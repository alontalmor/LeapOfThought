import random
import json
import logging

from tqdm import tqdm
import pandas as pd
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.common.file_utils import cached_path
from LeapOfThought.resources.wordnet import WordNet
from LeapOfThought.resources.conceptnet import ConceptNet

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TaxonomicReasoning(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'TaxonomicReasoning'
        logger.info("loading...")
        super().__init__(args)

        ### Loading conceptnet and filtering
        cn_edges_filt = ConceptNet().get_ConceptNet_full()
        cn_edges_filt = cn_edges_filt[
            (cn_edges_filt['relation'] != '/r/HasProperty') & \
            (cn_edges_filt['relation'] != '/r/ReceivesAction') & \
            (cn_edges_filt['relation'] != '/r/AtLocation') & \
            (cn_edges_filt['relation'] != '/r/UsedFor') & \
            (cn_edges_filt['relation'] != '/r/DerivedFrom') & \
            (cn_edges_filt['relation'] != '/r/RelatedTo') & \
            (cn_edges_filt['relation'] != '/r/FormOf') & \
            (cn_edges_filt['relation'] != '/r/Synonym') & \
            (cn_edges_filt['relation'] != '/r/SimilarTo') & \
            (cn_edges_filt['relation'] != '/r/MannerOf') & \
            (cn_edges_filt['relation'] != '/r/HasContext') & \
            (cn_edges_filt['relation'] != '/r/EtymologicallyRelatedTo') & \
            ~(cn_edges_filt['relation'].str.contains('/r/dbpedia'))]
        cn_edges_filt = cn_edges_filt.sort_values(by='weights').drop_duplicates(subset=['start_term', 'relation', 'end_term'], keep='last')
        end_term_filter = ['information', 'equipment', 'part', 'applicant', 'cloud thinking', 'state of matter', 'shape', 'drug', \
                           'plane figure', 'person', 'pets', 'food', 'assistant', 'make music', 'content', 'punishment', 'gathering', \
                           'wind instrument', 'unit', 'another name for homosexual', 'consequence', 'time period']
        cn_edges_filt = cn_edges_filt[~cn_edges_filt['end_term'].isin(end_term_filter)]
        self._ConceptNet_full = cn_edges_filt

        # Except for Desires and NotDesires we would like to have negative examples for capable of IsA, PartOf, and CapableOf
        # not the antonyms and distinct from are already negative of IsA, but the numbers are much less.
        self.ConcpetNet_Negative = self._ConceptNet_full[self._ConceptNet_full['relation'].isin(['/r/IsA', \
                                                                                                 '/r/PartOf', '/r/CapableOf'])].copy(
            deep=True)
        self.negative_class_relations = ['/r/IsA', '/r/PartOf', '/r/CapableOf']
        for relation in self.negative_class_relations:
            self.ConcpetNet_Negative.loc[self.ConcpetNet_Negative['relation'] == relation, 'end_term'] = \
                list(self.ConcpetNet_Negative.loc[self.ConcpetNet_Negative['relation'] == relation, 'end_term'].sample(frac=1,
                                                                                                                       random_state=0))
        self.ConcpetNet_Negative = self.ConcpetNet_Negative.set_index(['start_term', 'relation'])

        # loading a pre-made list of hypernyms
        wordnet_filtered_hypernyms_df = pd.read_csv('data/resources/wordnet_filtered_hypernyms.csv.gz', compression='gzip')
        terms_to_filter1 = ['action', 'message', 'artifact', 'quality', 'act', 'change', 'software', 'interface', 'painting', 'adult', \
                            'expert', 'worker', 'group', 'statement', 'function', 'happening', 'rate', 'shape', 'shrub', \
                            'informing', 'report', 'trait', 'restraint', 'information', 'helping', 'platform', 'appearance', 'tube', \
                            'property', 'disposition', 'trade', 'writing', 'communication', 'economy', 'system', 'practice', 'state', \
                            'work', 'degree', 'care', 'site', 'end', 'official', 'word', 'collection', 'people', 'body', 'feeling', 'part', \
                            'compound', 'condition', 'concept', 'representation', 'activity', 'area', 'space', 'region', 'layer',
                            'constituent', \
                            'object', 'structure', 'plant', 'organism', 'substance', 'organization', 'symbol', 'employee', 'organ', 'point', \
                            'professional', 'number', 'signal', 'implement']
        wordnet_filtered_hypernyms_df = wordnet_filtered_hypernyms_df[~wordnet_filtered_hypernyms_df['wordnet_synset'].str.contains('.v.')]
        wordnet_filtered_hypernyms_df = wordnet_filtered_hypernyms_df[~wordnet_filtered_hypernyms_df['end_term'].isin(terms_to_filter1)]
        wordnet_filtered_hypernyms_count = wordnet_filtered_hypernyms_df['end_term'].value_counts()
        wordnet_filtered_hypernyms_count = wordnet_filtered_hypernyms_count[wordnet_filtered_hypernyms_count > 100]
        self.selected_hypernyms = list(wordnet_filtered_hypernyms_count.index) + ['dog']

        # combining with the Aristo group hypernym list
        self.aristo_hypernyms_df = pd.read_csv('data/resources/taxonomy.txt', sep='\t', header=None, names=['subject', 'isA', 'hypernym'])
        # self.selected_hypernyms = list(set(self.selected_hypernyms) | set(self.aristo_hypernyms_df['hypernym']))

        self.hypernym_relation_filter = {'/r/CapableOf': ['machine', 'bird', 'drug', 'machine', 'person']}

        self.relation_templates_question = {
            '/r/IsA': ('Is [subject] a [object]?', 1),
            '/r/Antonym': ('Is [subject] a [object]?', 0),
            '/r/DistinctFrom': ('Is [subject] a [object]?', 0),
            '/r/PartOf': ('Is [subject] part of [object]?', 1),
            '/r/CapableOf': ('Is [subject] capable of [object]?', 1),
            '/r/Desires': ('Does [subject] desire [object]?', 1),
            '/r/NotDesires': ('Does [subject] desire [object]?', 0),
        }

        self.relation_templates_statement = {
            '/r/IsA': ('A [subject] is a [object].', 1),
            '/r/Antonym': ('A [subject] is a [object].', 0),
            '/r/DistinctFrom': ('A [subject] is a [object].', 0),
            '/r/PartOf': ('A [subject] is part of [object].', 1),
            '/r/CapableOf': ('A [subject] is capable of [object].', 1),
            '/r/Desires': ('A [subject] desires [object].', 1),
            '/r/NotDesires': ('A [subject] desires [object].', 0),
        }

        self.rules_templates = {
            '/r/IsA': 'If something is a [subject], then it is a [object].',
            '/r/Antonym': 'If something is a [subject], then it is not a [object].',
            '/r/DistinctFrom': 'If something is a [subject], then it is not a [object].',
            '/r/PartOf': 'If something is a [subject], then it is part of [object].',
            '/r/CapableOf': 'If something is a [subject], then it is capable of [object].',
            '/r/Desires': 'If something is a [subject], then it desires [object].',
            '/r/NotDesires': 'If something is a [subject], then it does not desire [object].',
        }

        self.negative_rules_templates = {
            '/r/IsA': 'If something is a [subject], then it is not a [object].',
            '/r/Antonym': 'If something is a [subject], then it is a [object].',
            '/r/DistinctFrom': 'If something is a [subject], then it is a [object].',
            '/r/PartOf': 'If something is a [subject], then it is not part of [object].',
            '/r/CapableOf': 'If something is a [subject], then it is not capable of [object].',
            '/r/Desires': 'If something is a [subject], then it does not desire [object].',
            '/r/NotDesires': 'If something is a [subject], then it desires [object].',
        }

    def find_hyponym_hypernym_pairs(self, hp):
        if self._config['debug']:
            print(f'---------------- {hp} ----------------------')
        conceptnet_hyponyms = self._ConceptNet_full[(self._ConceptNet_full['end_term'] == hp) &
                                                    (self._ConceptNet_full['relation'] == '/r/IsA') &
                                                    (self._ConceptNet_full['tf-start'] > 100000)]['start_term'].to_list()
        wordnet_hyponyms = WordNet().get_all_wordnet_hyponyms(hp, ['01'], levels_down=9, output_type='names')
        aristo_hyponyms = self.aristo_hypernyms_df[self.aristo_hypernyms_df['hypernym'] == hp]['subject']
        all_hyponyms = sorted(list((set(conceptnet_hyponyms) & set(wordnet_hyponyms)) | set(aristo_hyponyms)))
        return [(h, hp) for h in all_hyponyms]

    def test_hypernym_model_knowledge_add_hp_examples(self, hyponym_hypernym_pairs, hp, split):
        for pair in hyponym_hypernym_pairs:
            example = {'phrase': pair[0] + ' is a ' + pair[1], \
                       'subject': pair[0],
                       'hypernym': hp,
                       'answer': 1,
                       'split': split}

            # append_teachyourai_format_example() is method implemented in ArtiSet class and takes an example dict
            # (that must contain a "phrase", "answer") and converts it to a BooleanQA format
            self.append_teachyourai_format_example(example, do_print=False)
            self.examples_meta.append(example)

    def build_artificial_dataset(self, args):
        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []
        random.seed(17)

        logger.info("building examples")

        # Find good hypernym candidates (fixed list for now)
        # hypernyms = ['dog','insect','bird','food','animal','mammal']

        # For each hypernym candidate we would like to create a list of it's hyponyms from
        # different level in the taxonomic tree.
        example_relations = []
        for hp in tqdm(self.selected_hypernyms):

            hyponym_hypernym_pairs = self.find_hyponym_hypernym_pairs(hp)

            # Choosing the split using the hypernym value
            if hp in ['animal', 'mammal', 'flower', 'fruit', 'music']:
                split = 'dev'
            else:
                split = 'train'

            if args.variant == 'hypernym_model_knowledge':
                self.test_hypernym_model_knowledge_add_hp_examples(hyponym_hypernym_pairs, hp, split)
                continue

            hypernym_rules = self._ConceptNet_full[(self._ConceptNet_full['start_term'] == hp) & \
                                                   (self._ConceptNet_full['weights'] > 1)]

            for id, p in hypernym_rules.iterrows():
                if p['relation'] not in self.relation_templates_statement or \
                        (p['relation'] in self.hypernym_relation_filter and hp in self.hypernym_relation_filter[p['relation']]):
                    # logger.error(p['relation'])
                    continue

                if len(hypernym_rules) > 100 and len(hyponym_hypernym_pairs) > 7:
                    hyponym_hypernym_pairs_for_rule = random.sample(hyponym_hypernym_pairs, 7)
                else:
                    hyponym_hypernym_pairs_for_rule = hyponym_hypernym_pairs

                if self._config['debug'] and len(hyponym_hypernym_pairs_for_rule) > 2:
                    [print(h + (p['relation'].replace('/r/', ''), p['end_term'])) \
                     for h in random.sample(hyponym_hypernym_pairs_for_rule, 2)]

                for pair in hyponym_hypernym_pairs_for_rule:
                    # We may have more than one variant to our artiset
                    # if args.variant in ['only_question','hypernym_implicit']:

                    if p['relation'] in self.negative_class_relations:
                        examples_to_gen = ['positive', 'negative']
                    else:
                        examples_to_gen = ['positive']

                    for example_type in examples_to_gen:
                        if example_type == 'negative':
                            end_term = random.sample(list(self.ConcpetNet_Negative.loc[(p['start_term'], \
                                                                                        p['relation'])]['end_term']), 1)[0]
                            question = self.relation_templates_statement[p['relation']][0] \
                                .replace('[subject]', pair[0]).replace('[object]', end_term)
                            answer = 1 - self.relation_templates_statement[p['relation']][1]
                            rules_templatesrules_templates = self.negative_rules_templates
                            opposite_rules_templates = self.rules_templates

                        else:
                            end_term = p['end_term']
                            question = self.relation_templates_statement[p['relation']][0] \
                                .replace('[subject]', pair[0]).replace('[object]', end_term)
                            answer = self.relation_templates_statement[p['relation']][1]
                            rules_templates = self.rules_templates
                            opposite_rules_templates = self.negative_rules_templates

                        if args.variant == 'training_mix':
                            # We are not using numpy to preserve the random.seed()
                            variant = random.sample(['hypernym_implicit_with_distractors'] * 4 + \
                                                    ['same_obj_rule_with_distractors'] +
                                                    ['only_distractors'], 1)[0]
                        else:
                            variant = args.variant

                        context = ''
                        distractors = []
                        if variant in ['hypernym_implicit']:
                            context = rules_templates[p['relation']].replace('[subject]', pair[1]).replace('[object]', end_term)
                        elif variant in ['hypernym_implicit_with_distractors']:
                            # adding an opposite rule with the same relation on subject, using a different HP
                            context = rules_templates[p['relation']].replace('[subject]', pair[1]).replace('[object]', end_term)
                            # for i in range(random.sample([1,2,3])):
                            dist_hp = random.sample(list(set(self.selected_hypernyms) - set(hp[1])), 1)[0]
                            ### TODO get rid of this.
                            distractors.append(opposite_rules_templates[p['relation']].replace('[subject]', \
                                                                                               dist_hp).replace('[object]', end_term))
                        elif variant in ['only_distractors']:
                            context = ''
                        elif variant in ['same_obj_rule_with_distractors']:
                            # adding an opposite rule with the same relation on subject, using a different HP
                            context = rules_templates[p['relation']].replace('[subject]', pair[0]).replace('[object]', end_term)
                            # for i in range(random.sample([1,2,3])):
                            dist_hp = random.sample(list(set(self.selected_hypernyms) - set(hp[1])), 1)[0]
                            distractors.append(opposite_rules_templates[p['relation']].replace('[subject]', \
                                                                                               dist_hp).replace('[object]', end_term))

                        elif variant == 'same_obj_rule':
                            context = rules_templates[p['relation']].replace('[subject]', pair[0]).replace('[object]', end_term)
                        elif variant == 'co-hyponym_distractor':
                            context = rules_templates[p['relation']].replace('[subject]', \
                                                                             random.sample([p[0] for p in hyponym_hypernym_pairs], 1)[
                                                                                 0]).replace('[object]', end_term)
                        elif variant == 'hypernym_explicit':
                            context = 'A ' + pair[0] + ' is a ' + pair[1] + '. ' + \
                                      rules_templates[p['relation']].replace('[subject]', pair[1]).replace('[object]', end_term)
                        elif variant == 'hypernym_implicit_counterfactual':
                            if rules_templates != self.negative_rules_templates:
                                context = self.negative_rules_templates[p['relation']].replace('[subject]', \
                                                                                               pair[1]).replace('[object]', end_term)
                                answer = 1 - answer
                            else:
                                context = self.rules_templates[p['relation']].replace('[subject]', \
                                                                                      pair[1]).replace('[object]', end_term)
                                answer = 1 - answer

                        example = {'phrase': question, \
                                   'answer': answer,
                                   'hypernym': hp,
                                   'subject': pair[0],
                                   'relation': p['relation'],
                                   'object': end_term,
                                   'context': context,
                                   'split': split,
                                   'distractors': distractors}

                        # append_teachyourai_format_example() is method implemented in ArtiSet class and takes an example dict
                        # (that must contain a "phrase", "answer") and converts it to a BooleanQA format
                        self.append_teachyourai_format_example(example, do_print=False)
                        self.examples_meta.append(example)

        self.examples_meta = pd.DataFrame(self.examples_meta)
        if args.variant in ['hypernym_implicit_with_distractors', 'training_mix']:
            self.examples_meta = self.examples_meta.set_index('id')
            for example in self.artiset_data:
                if len(example['context']) > 0:
                    context_list = [example['context']]
                else:
                    context_list = []
                context_list += example['distractors']
                random_distractors = list(self.examples_meta[self.examples_meta['split'] == example['split']]['context'])
                context_list += random.sample(random_distractors, random.sample([0, 1, 2, 3, 4, 5], 1)[0])

                random.shuffle(context_list)
                example['context'] = ' '.join(context_list)
            for example in self.artiset_data:
                self.examples_meta.loc[example['id'], 'context'] = example['context']
            self.examples_meta = self.examples_meta.reset_index()

        logger.info(f"top 20 question per hypernym:\n {self.examples_meta['hypernym'].value_counts()[0:20]}")
        # save_dataset() is a is method implemented in ArtiSet class that automatically saves the artiset
        # if the config output_file contains the string _sample.jsonl it will be saved in a more readable format
        # otherwise it will split the examples in self.artiset_data into train, dev, test and save them in s3
        # if output_file startswith s3:// otherwise locally. (If output_file is empty, it will not save)
        self.save_dataset()