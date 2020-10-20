
import random, copy
import logging
import pandas as pd
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.resources.teachai_kb import TeachAIKB

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

class Meronyms(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'Meronyms'
        self.variant = args.variant
        logger.info("loading...")
        super().__init__(args)

    def build_artificial_dataset(self, args):
        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []
        random.seed(17)

        logger.info("building examples")

        # Find good meronym candidates (fixed list for now)
        meronyms = {}
        #devset_subjects = [e['subject'] for e in TeachAIKB().sample({'predicate': ['hypernym'], \
        #                    'object': ['food', 'vehicle', 'road', 'clothing', \
        #                               'instrument','commodity','activity'], 'validity': ['always true']})]

        # Sampling from meronyms in our dataset that are true
        meronyms['dev'] = TeachAIKB().sample({'predicate':['meronym'],
                                              'object_not_in': ['head','storey','sauce','face','room','blossom',\
                                                                'bedroom','sandwich','skull','doorway','hull'],
                                              'validity':['always true']}, sample_limit=('object', 70), tar_tag='implicit_rule')
        #meronyms['train'] = TeachAIKB().sample({'predicate': ['meronym', 'part of'],
        #                                        #'subject_not_in': devset_subjects,
        #                                        'validity': ['always true']}, sample_limit=('object', 70), tar_tag='implicit_rule')

        for split, bridge_rules in meronyms.items():
            if args.split is not None and split != args.split:
                continue

            logger.info(f'---------------  {split} ------------------')
            # To have only true rules, "never true" will be applied negative language: "never true" + negative = "always true"
            # Adding a property for each meronym object ("animal") --> "animals are capable of living"
            meronym_property_positive = TeachAIKB().connect(connect_to=bridge_rules, max_to_connect = 5,
                                    constraints={'validity': 'always true', 'predicate': ['meronym','part of']}, \
                                    src_tags=['implicit_rule'], connection_point=[{'object':'subject'}], tar_tag='property')

            # now that we have positive example we will try to create negative examples that mimic the distribution
            # of the positive ones.
            #meronym_property_negative = TeachAIKB().connect_negative_shuffle_subject(shuffle=meronym_property_positive, \
            #                        shuffle_on='property', tar_tag='property', avoid_mixing=['co-meronyms'])
            meronym_property_negative = self.self_negative_subject_sample(meronym_property_positive, sample_on='property', \
                                         avoid_mixing=['co-meronyms','hyponyms'])

            # creating a statement by applying downward monotonicity to the meronym and it's property.
            examples = TeachAIKB().connect_downward_monotone(connect_to=meronym_property_positive + meronym_property_negative,
                                    scope='implicit_rule', property='property', tar_tag='statement')

            # Sampling distractors.  "('statement', ['predicate', 'object'], 2)" means for each example, sample at most two statements
            # with the same ['predicate', 'object'] as the example statement and add to distractors.
            self.sample_distractors(examples, sample=[
                ('property', ['predicate','object'], 2, True, ['implicit_rule','property'], \
                    True if args.variant != 'statement_only' else False),
                ('statement', ['subject'], 2, False, ['statement'], False),
                ('statement', ['predicate'], 2, False, ['statement'],False)], tar_tag='distractors')
            if True:  # Change condition to config flag if need be
                # Change implicit distractor to be the negative statement for main subject
                # E.g., instead of "salmon has eye" distractor for "house has door", we make it "house does not have eye"
                for e in examples:
                    dist = copy.deepcopy(e['distractors']['implicit_rule'][0])
                    dist['subject'] = e['implicit_rule']['subject']
                    dist['validity'] = 'never true'
                    e['distractors']['implicit_rule'] = [dist]

            # mixing 10% of the implicit rule as statement in the training mix only.
            if args.variant == 'training_mix':
                examples += [{'statement': e['implicit_rule']} for e in random.sample(bridge_rules, int(0.05 * float(len(examples))))]
                negative_bridge = self.self_negative_subject_sample(bridge_rules, sample_on='implicit_rule', avoid_mixing=['co-meronyms'])
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
            else:
                self.build_statement_rule_property_examples(examples, split=split, ablation_list=ablations[args.variant])


        self.print_examples(20)
        self.print_stats()
        self.examples_meta = pd.DataFrame(self.examples_meta)
        self.save_dataset()






