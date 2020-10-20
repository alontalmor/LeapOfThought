
import random
import copy
import logging

from tqdm import tqdm
import pandas as pd
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.common.file_utils import cached_path
from LeapOfThought.resources.wordnet import WordNet
from LeapOfThought.resources.conceptnet import ConceptNet
from LeapOfThought.resources.teachai_kb import TeachAIKB
from LeapOfThought.common.data_utils import uniform_sample_by_column, pandas_multi_column_agg

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

class Hypernyms(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'Hypernyms'
        self.variant = args.variant
        logger.info("loading...")
        super().__init__(args)

    def build_artificial_dataset(self, args):
        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []
        random.seed(17)

        logger.info("building examples")

        # Find good hypernym candidates (fixed list for now)
        hypernyms = {}
        dev_objects = ['tree', 'flower', 'fruit', 'music', 'bird', 'alcohol', 'plant']
        # Sampling from hypernyms in our dataset that are true
        hypernyms['dev'] = TeachAIKB().sample({'predicate':['hypernym'], 'source_not_in': ['wikidata'],
                                               'object': dev_objects,
                                               'validity':['always true']}, tar_tag='implicit_rule')
        hypernyms['train'] = TeachAIKB().sample({'predicate': ['hypernym'], 'source_not_in': ['wikidata'],
                                                 'object_not_in': dev_objects,
                                                 'validity': ['always true']}, tar_tag='implicit_rule')

        for split, bridge_rules in hypernyms.items():
            if args.split is not None and split != args.split:
                continue

            logger.info(f'---------------  {split} ------------------')
            # To have only true rules, "never true" will be applied negative language: "never true" + negative = "always true"
            # Adding a property for each hypernym object ("animal") --> "animals are capable of living"
            hypernym_property_positive = TeachAIKB().connect(connect_to=bridge_rules, max_to_connect = 5,
                                    constraints={'validity': 'always true', 'predicate_not_in': ['hypernym']}, \
                                    src_tags=['implicit_rule'], connection_point=[{'object':'subject'}], tar_tag='property')

            # now that we have positive example we will try to create negative examples that mimic the distribution
            # of the positive ones.
            hypernym_property_negative = self.self_negative_subject_sample(hypernym_property_positive, sample_on='property',\
                                                                           avoid_mixing='hyponyms')

            # creating a statement by applying downward monotonicity to the hypernym and it's property.
            examples = TeachAIKB().connect_downward_monotone(connect_to=hypernym_property_positive + hypernym_property_negative,
                                    scope='implicit_rule', property='property', tar_tag='statement')

            # Sampling distractors.  "[(src_tag, src_fields, num_to_sample, exactly_sample_num, fields_to_take, balance_with_statement)]
            self.sample_distractors(examples, tar_tag='distractors', sample=[
                ('property', ['predicate', 'object'], 2, True, ['implicit_rule', 'property'], True if args.variant != 'statement_only' else False),
                ('statement', ['predicate'], 2, False, ['statement'], False),
                ('statement', ['subject'], 2, False, ['statement'], False)])
            if True:  # Change condition to config flag if need be
                # Change implicit distractor to be the negative statement for main subject
                # E.g., instead of "salmon is fish" distractor for "whale is mammal", we make it "whale is not fish"
                for e in examples:
                    dist = copy.deepcopy(e['distractors']['implicit_rule'][0])
                    dist['subject'] = e['implicit_rule']['subject']
                    dist['validity'] = 'never true'
                    e['distractors']['implicit_rule'] = [dist]

            # mixing 10% of the implicit rule as statement in the training mix only.
            if args.variant == 'training_mix' and split == 'train':
                examples += [{'statement': e['implicit_rule']} for e in random.sample(bridge_rules, int(0.05 * float(len(examples))))]
                negative_bridge = self.self_negative_subject_sample(bridge_rules, sample_on='implicit_rule',
                                                                    avoid_mixing=['hyponyms'])
                examples += [{'statement': e['implicit_rule']} for e in random.sample(negative_bridge, int(0.05 * float(len(examples))))]

            # for each variation, the proportion in which each rule type will be filtered.
            ablations = {
                'training_mix': [(['implicit_rule'], 0.5), (['property'], 0), (['distractors'], 0.2)],
                'statement_only': [(['implicit_rule'], 1), (['property'], 1)],
                'explicit_only': [(['implicit_rule'], 0), (['property'], 0)],
                'corrected_only': [(['implicit_rule'], 0), (['property'], 0)],
                'corrected_inv_only': [(['implicit_rule'], 0), (['property'], 0)],
                'implicit_only': [(['implicit_rule'], 1), (['property'], 0)],
                'statement_only_no_context': [(['implicit_rule'], 1), (['property'], 1), (['distractors'], 1)],
                'statement_subject_lang_selectivity': [(['implicit_rule'], 1), (['property'], 0)],
                'implicit_knowledge_test': [(['statement'], 1), (['implicit_rule'], 0), (['property'], 1), (['distractors'], 1)],
                'implicit_knowledge_distractor_test': [(['statement'], 1), (['implicit_rule'], 1), (['property'], 1), (['distractors'], 1)],
            }

            if args.variant == "corrected_only" or args.variant == "corrected_inv_only":
                # In these variants we update the examples to delete implicit_rule edges according to
                # absence ("corrected_only") or presence ("corrected_inv_only") in self._incorrect_beliefs
                # So "corrected_only" means that context include implicit rules that a model got wrong.
                def edge_tuple(e):
                    return (e['subject'], e['predicate'], e['object'], e['validity'])
                incorrect_beliefs_set = {edge_tuple(e) for e in self._incorrect_beliefs}

                def delete_edge(e):
                    tuple = edge_tuple(e)
                    if args.variant == "corrected_only":
                        return tuple not in incorrect_beliefs_set
                    else:
                        return tuple in incorrect_beliefs_set

                for e in examples:
                    if delete_edge(e['implicit_rule']):
                        del e['implicit_rule']
                    dist = e['distractors']
                    if delete_edge(dist['implicit_rule'][0]):
                        del dist['implicit_rule']

            if args.variant == 'implicit_knowledge_test':
                self.build_statement_rule_property_examples(examples, split=split, statement_tag='implicit_rule', rule_tags=[],
                                                            distractor_tags=[])
            elif args.variant == 'implicit_knowledge_distractor_test':
                for e in examples:
                    e['distractor_implicit_rule'] = copy.deepcopy(e['implicit_rule'])
                    e['distractor_implicit_rule']['object'] = e['distractors']['implicit_rule'][0]['object']
                    e['distractor_implicit_rule']['validity'] = 'never true'
                self.build_statement_rule_property_examples(examples, split=split, statement_tag='distractor_implicit_rule', rule_tags=[],
                                                            distractor_tags=[])

            # Actively splitting between test and dev (50/50)
            if split == 'dev':
                # making sure that for the same amount of examples the split will always be the same.
                random.seed(17)
                all_inds = [ i for i in range(len(examples))]
                dev_inds = random.sample(all_inds, int(len(all_inds) / 2))
                test_inds = list(set(all_inds) - set(dev_inds))
                splits = [('dev', [examples[i] for i in dev_inds]),
                          ('test', [examples[i] for i in test_inds])]
            else:
                splits = [('train', examples )]

            for final_split, final_examples in splits:
                if args.variant == 'implicit_knowledge_test':
                    self.build_statement_rule_property_examples(final_examples, split=final_split, statement_tag='implicit_rule', rule_tags=[],
                                                                distractor_tags=[])
                elif args.variant == 'implicit_knowledge_distractor_test':
                    for e in final_examples:
                        e['distractor_implicit_rule'] = copy.deepcopy(e['implicit_rule'])
                        e['distractor_implicit_rule']['object'] = e['distractors']['implicit_rule'][0]['object']
                        e['distractor_implicit_rule']['validity'] = 'never true'
                    self.build_statement_rule_property_examples(final_examples, split=final_split, statement_tag='distractor_implicit_rule', rule_tags=[],
                                                                distractor_tags=[])
                else:
                    self.build_statement_rule_property_examples(final_examples, split=final_split, ablation_list=ablations[args.variant])

        self.print_examples(20)
        self.print_stats()
        self.examples_meta = pd.DataFrame(self.examples_meta)
        self.save_dataset()






