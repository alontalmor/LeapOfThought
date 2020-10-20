import pandas as pd
import requests
import os
from LeapOfThought.common.file_utils import cached_path
from tqdm import tqdm
import pandas as pd
import copy
import random
from LeapOfThought.resources.wordnet import WordNet
from LeapOfThought.resources.conceptnet import ConceptNet
from LeapOfThought.resources.wikidata import WikiData
from LeapOfThought.common.data_utils import uniform_sample_by_column, pandas_multi_column_agg

# This is mainly for testing and debugging  ...
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 130)
pd.set_option("display.colheader_justify", "left")

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
#logger.setLevel( logging.DEBUG )

class TeachAIKB:
    """ A python singleton """

    class __impl:
        """ Implementation of the singleton interface """

        def __init__(self):
            self.data_path = 'data/resources/teachai_kb.jsonl.gz'

            if os.path.exists('data/resources/teachai_kb.csv'):
                self.kb = pd.read_csv('data/resources/teachai_kb.csv') \
                # creating index:
                self.kb_full_index = self.kb.set_index(['subject','predicate','object'])
                self.predicate_object_index = self.kb.set_index(['predicate', 'object'])
                self.subject_predicate_index = self.kb.set_index(['subject','predicate'])
                self.kb = self.kb.to_dict(orient='rows')
            else:
                logger.info('data/resources/teachai_kb.csv not found, please build it')


            ### subject sources

            # people names
            self.person_names = []
            with open('data/resources/male_names.txt') as f:
                self.person_names += [name.replace('\n', '') for name in f]
            with open('data/resources/female_names.txt') as f:
                self.person_names += [name.replace('\n', '') for name in f]

            self.relation_templates_statement = {
                '/r/IsA': 'A [subject] is a [object].',
                '/r/Antonym': 'A [subject] is the opposite of  [object].',
                '/r/DistinctFrom': 'A [subject] is distinct from [object].',
                '/r/PartOf': 'A [subject] is part of [object].',
                '/r/CapableOf': 'A [subject] is capable of [object].',
                '/r/Desires': 'A [subject] desires [object].',
                '/r/NotDesires': 'A [subject] does not desire [object].',
                'meronym': 'A [subject] has a [object].',
                'hypernym': 'A [subject] is a [object].',
                'part of': 'A [subject] has a [object]. (CN)',

            }

            self.relation_templates_statement_shorthand = {'super bowl winner': ['[subject] lost the Super Bowl of [object].',
                                                                    'The Super Bowl of [object] was lost by [subject].'],
                                              'super bowl loser': ['[subject] won the Super Bowl of [object].',
                                                                   'The Super Bowl of [object] was won by [subject].'],
                                              'band member': ['[subject] is a member of [object].',
                                                              '[object] has [subject] as a member.',
                                                              "[subject] is [object]'s member."],
                                              'capital': ['[subject] is the capital of [object].',
                                                          'The capital of [object] is [subject].',
                                                          "[object]'s capital is [subject]."],
                                              'director': ['[subject] is director of [object].', "[object]'s director is [subject]."],
                                              'release year': ['[subject] is the release year of [object].',
                                                               '[object] was released on the year [subject].'],
                                              'founder': ['[subject] founded [object].', '[object] was founded by [subject].'],
                                              'headquarter': ['[subject] is the headquarters of [object].',
                                                              "[object]'s headquarters is [subject]."],
                                              'child': ['[subject] is the child of [object].', "[object]'s child is [subject]."],
                                              'spouse': ['[subject] is the spouse of [object].', "[object] is [subject]'s spouse ."],
                                              'CEO': ['[subject] is the CEO of [object].', "[object]'s CEO is [subject]."]}

            self.relation_templates_negative_statement_shorthand = {
                                               'super bowl winner': ['[subject] did not lose the Super Bowl of [object].',
                                                                     "The Super Bowl of [object] wasn't lost by [subject]."],
                                               'super bowl loser': ["[subject] did not win the Super Bowl of [object].",
                                                                    "The Super Bowl of [object] wasn't won by [subject]."],
                                               'band member': ['[subject] is not a member of [object].',
                                                               '[object] does not have [subject] as a member.',
                                                               "[subject] isn't [object]'s member."],
                                               'capital': ['[subject] is not the capital of [object].',
                                                           'The capital of [object] is not [subject].',
                                                           "[object]'s capital isn't [subject]."],
                                               'director': ['[subject] is not director of [object].',
                                                            "[object]'s director isn't [subject]."],
                                               'release year': ['[subject] is not the release year of [object].',
                                                                '[object] was not released on the year [subject].'],
                                               'founder': ['[subject] did not found [object].', "[object] wasn't founded by [subject]."],
                                               'headquarter': ['[subject] is not the headquarters of [object].',
                                                               "[object]'s headquarters isn't [subject]."],
                                               'child': ['[subject] is not the child of [object].',
                                                         "[object]'s child isn't [subject]."],
                                               'spouse': ['[subject] is not the spouse of [object].',
                                                          "[object] isn't [subject]'s spouse ."],
                                               'CEO': ['[subject] is not the CEO of [object].', "[object]'s CEO isn't [subject]."]}

            self.rules_templates_long = {
                '/r/IsA': 'If something is a [subject], then it is a [object].',
                '/r/Antonym': 'If something is a [subject], then it is not a [object].',
                '/r/DistinctFrom': 'If something is a [subject], then it is not a [object].',
                '/r/PartOf': 'If something is a [subject], then it is part of [object].',
                '/r/CapableOf': 'If something is a [subject], then it is capable of [object].',
                '/r/Desires': 'If something is a [subject], then it desires [object].',
                '/r/NotDesires': 'If something is a [subject], then it does not desire [object].',
                'hypernym': 'If something is a [subject], then it is a [object].',
                'meronym': 'If something is a [subject], then it has a [object].',
                'part of': 'If something is a [subject], then it has a [object]. (CN)',
            }

            self.rules_templates = {
                '/r/IsA': 'A [subject] is a [object].',
                '/r/Antonym': 'A [subject] is not a [object].',
                '/r/DistinctFrom': 'A [subject] is not a [object].',
                '/r/PartOf': 'A [subject] is part of [object].',
                '/r/CapableOf': 'A [subject] is capable of [object].',
                '/r/Desires': 'A [subject] desires [object].',
                '/r/NotDesires': 'A [subject] does not desire [object].',
                'hypernym': 'A [subject] is a [object].',
                'meronym': 'A [subject] has a [object].',
                'part of': 'A [subject] has a [object]. (CN)',
            }

            self.rules_templates_shorthand = {'super bowl winner': ['[subject] lost the Super Bowl of [object].',
                                                                    'The Super Bowl of [object] was lost by [subject].'],
                                              'super bowl loser': ['[subject] won the Super Bowl of [object].',
                                                                   'The Super Bowl of [object] was won by [subject].'],
                                              'band member': ['[subject] is a member of [object].',
                                                              '[object] has [subject] as a member.',
                                                              "[subject] is [object]'s member."],
                                              'capital': ['[subject] is the capital of [object].',
                                                          'The capital of [object] is [subject].',
                                                          "[object]'s capital is [subject]."],
                                              'director': ['[subject] is director of [object].', "[object]'s director is [subject]."],
                                              'release year': ['[subject] is the release year of [object].',
                                                               '[object] was released on the year [subject].'],
                                              'founder': ['[subject] founded [object].', '[object] was founded by [subject].'],
                                              'headquarter': ['[subject] is the headquarters of [object].',
                                                              "[object]'s headquarters is [subject]."],
                                              'child': ['[subject] is the child of [object].', "[object]'s child is [subject]."],
                                              'spouse': ['[subject] is the spouse of [object].', "[object] is [subject]'s spouse ."],
                                              'CEO': ['[subject] is the CEO of [object].', "[object]'s CEO is [subject]."],
                                              'has 1': ['[subject] has one [object].', 'Only one [object] is what [subject] has.',
                                                        '[subject] number of [object]s is one.'],
                                              'has 2': ['[subject] has two [object]s.', 'Only two [object]s is what [subject] has.',
                                                        '[subject] number of [object]s is two.'],
                                              'has 3': ['[subject] has three [object]s.', 'Only three [object]s is what [subject] has.',
                                                        '[subject] number of [object]s is three.'],
                                              'has 4': ['[subject] has four [object]s.', 'Only four [object]s is what [subject] has.',
                                                        '[subject] number of [object]s is four.'],
                                              'has 5': ['[subject] has five [object]s.', 'Only five [object]s is what [subject] has.',
                                                        '[subject] number of [object]s is five.'],
                                              'hypernym': '[subject] is a [object].'}

            self.negative_rules_templates_long = {
                '/r/IsA': 'If something is a [subject], then it is not a [object].',
                '/r/Antonym': 'If something is a [subject], then it is a [object].',
                '/r/DistinctFrom': 'If something is a [subject], then it is a [object].',
                '/r/PartOf': 'If something is a [subject], then it is not part of [object].',
                '/r/CapableOf': 'If something is a [subject], then it is not capable of [object].',
                '/r/Desires': 'If something is a [subject], then it does not desire [object].',
                '/r/NotDesires': 'If something is a [subject], then it desires [object].',
                'hypernym': 'If something is a [subject], then it is not a [object].',
                'meronym': 'If something is a [subject], then it does not have a [object].',
                'part of': 'If something is a [subject], then it does not have a [object]. (CN)',
            }

            self.negative_rules_templates = {
                '/r/IsA': 'A [subject] is not a [object].',
                '/r/Antonym': 'A [subject] is a [object].',
                '/r/DistinctFrom': 'A [subject] is a [object].',
                '/r/PartOf': 'A [subject] is not part of [object].',
                '/r/CapableOf': 'A [subject] is not capable of [object].',
                '/r/Desires': 'A [subject] does not desire [object].',
                '/r/NotDesires': 'A [subject] desires [object].',
                'hypernym': 'A [subject] is not a [object].',
                'meronym': 'A [subject] does not have a [object].',
                'part of': 'A [subject] does not have a [object]. (CN)',
            }

        def construct_kb(self):
            self.kb = []
            logger.info('Adding Funcational relations from WikiData')
            self.add_func_rel_from_wikidata()
            logger.info('Loading and preprocessing ConceptNet')
            self.load_concpetnet()
            logger.info('Adding myronyms')
            self.add_oyvind_meronyms()
            logger.info('Adding hypernyms')
            self.add_hypernyms()
            logger.info('Adding always true ConceptNet properties')
            self.add_always_true_conceptnet_properties()

            #logger.info('Adding ConceptNet part of as meronyms')
            #self.add_conceptnet_part_of()


            # filtering bad triplets
            self.filter_bad_entries()
            #logger.info('Adding never true')
            #self.add_never_true()

            kb_df = pd.DataFrame(self.kb).sample(frac=1)

            # removing non pc words
            self._non_pc_words = pd.read_csv('data/resources/pc_words.csv')['words'].to_list()
            kb_df = kb_df[~kb_df['subject'].isin(self._non_pc_words)]
            kb_df = kb_df[~kb_df['object'].isin(self._non_pc_words)]

            # overriding with manual rules. First merge all the rules (the manual ones will be last)
            kb_df = pd.merge(kb_df, pd.read_csv('data/resources/manual_kb.csv'), \
                             on=['subject', 'predicate', 'object', 'source','validity'], how="outer")
            # now drop duplicates. This way manual rules are always added or override existing rules.
            # The way to delete a rule, is to just change it's validity (to say "always false")
            kb_df = kb_df.drop_duplicates(subset=['subject', 'predicate', 'object'], keep='last')
            kb_df = kb_df[(kb_df['subject'].notnull()) & (kb_df['object'].notnull())]
            kb_df = kb_df[(kb_df['subject'].astype(str).apply(len) > 0) & \
                          (kb_df['object'].astype(str).apply(len) > 0)]

            logger.info(f'DONE. total of {len(kb_df)} triplets saved.')
            logger.info(f"Predicate distribution:\n{kb_df['predicate'].value_counts()}")

            kb_df.to_csv('data/resources/teachai_kb.csv', index=False)

            ## animal names (dogs cats etc... )

            ### Properties
            # colors
            # qualities (nice)
            # physical (fat slim , tall etc.. )

            ### Possession
            # has tv ...

            ### Hypernyms (types)

            # person, animals taxonomy, plants taxonomy,

            ### locations?

            ### actions?

            ### cause and effect / timely?

        def filter_bad_entries(self):

            filtered_kb = []
            for edge in self.kb:
                # we do not support self edge
                if edge['subject'] == edge['object']:
                    continue

                filtered_kb.append(edge)
            self.kb = filtered_kb

        def load_concpetnet(self):
            ### Loading conceptnet and filtering
            cn_edges_filt = ConceptNet().get_ConceptNet_full()
            cn_edges_filt = cn_edges_filt[
                (cn_edges_filt['relation'] != '/r/DerivedFrom') & \
                (cn_edges_filt['relation'] != '/r/RelatedTo') & \
                (cn_edges_filt['relation'] != '/r/FormOf') & \
                (cn_edges_filt['relation'] != '/r/Synonym') & \
                (cn_edges_filt['relation'] != '/r/SimilarTo') & \
                (cn_edges_filt['relation'] != '/r/MannerOf') & \
                (cn_edges_filt['relation'] != '/r/HasContext') & \
                (cn_edges_filt['relation'] != '/r/EtymologicallyRelatedTo') & \
                ~(cn_edges_filt['relation'].str.contains('/r/dbpedia'))]
            cn_edges_filt = cn_edges_filt.sort_values(by='weights').drop_duplicates(subset=['start_term', 'relation', 'end_term'],
                                                                                    keep='last')
            self._ConceptNet_full = cn_edges_filt

        def add_hypernyms(self):
            # loading a pre-made list of hypernyms
            wordnet_filtered_hypernyms_df = pd.read_csv('data/resources/wordnet_filtered_hypernyms.csv.gz', compression='gzip')
            filt1 = ['action', 'message', 'artifact', 'quality', 'act', 'change', 'software', 'interface', 'painting', 'adult', \
                     'expert', 'worker', 'group', 'statement', 'function', 'happening', 'rate', 'shape', 'shrub', \
                     'informing', 'report', 'trait', 'restraint', 'information', 'helping', 'platform', 'appearance', 'tube', \
                     'property', 'disposition', 'trade', 'writing', 'communication', 'economy', 'system', 'practice', 'state', \
                     'work', 'degree', 'care', 'site', 'end', 'official', 'word', 'collection', 'people', 'body', 'feeling',
                     'part', 'compound', 'condition', 'concept', 'representation', 'activity', 'area', 'space', 'region', 'layer',
                     'constituent', 'object', 'structure', 'plant', 'organism', 'substance', 'organization', 'symbol', 'employee', 'organ',
                     'point', 'professional', 'number', 'signal', 'implement']
            wordnet_filtered_hypernyms_df = wordnet_filtered_hypernyms_df[
                ~wordnet_filtered_hypernyms_df['wordnet_synset'].str.contains('.v.')]
            wordnet_filtered_hypernyms_df = wordnet_filtered_hypernyms_df[~wordnet_filtered_hypernyms_df['end_term'].isin(filt1)]
            wordnet_filtered_hypernyms_count = wordnet_filtered_hypernyms_df['end_term'].value_counts()
            wordnet_filtered_hypernyms_count = wordnet_filtered_hypernyms_count[wordnet_filtered_hypernyms_count > 100]
            selected_hypernyms = list(wordnet_filtered_hypernyms_count.index) + ['dog']

            aristo_hypernyms_df = pd.read_csv('data/resources/taxonomy.txt', sep='\t', header=None,
                                                   names=['subject', 'isA', 'hypernym'])

            pbar = tqdm(desc='Hypernyms added')
            for hp in selected_hypernyms:
                conceptnet_hyponyms = self._ConceptNet_full[(self._ConceptNet_full['end_term'] == hp) &
                                                            (self._ConceptNet_full['relation'] == '/r/IsA') &
                                                            (self._ConceptNet_full['tf-start'] > 100000)]['start_term'].to_list()
                wordnet_hyponyms = WordNet().get_all_wordnet_hyponyms(hp, ['01'], levels_down=9, output_type='names')
                aristo_hyponyms = aristo_hypernyms_df[aristo_hypernyms_df['hypernym'] == hp]['subject']
                all_hyponyms = sorted(list((set(conceptnet_hyponyms) & set(wordnet_hyponyms)) | set(aristo_hyponyms)))
                # TODO add synsets / lemmas / TFs
                for h in all_hyponyms:
                    pbar.update(1)
                    self.kb.append({'subject':h,
                                    'predicate':'hypernym',
                                    'object':hp,
                                    'source': 'wordnet/conceptnet',
                                    'validity':'always true'})
            pbar.close()

        def add_oyvind_meronyms(self):
            isa_parts_v3 = pd.read_csv('data/resources/wordnet-simple-isa-parts-v3.tsv.zip', compression='zip', sep='\t',
                                                      names=['subject', 'predicate', 'object'])
            isa_parts_v3['predicate'] = isa_parts_v3['predicate'].str.replace('wnsimple_hypernym_direct', 'hypernym')
            isa_parts_v3['predicate'] = isa_parts_v3['predicate'].str.replace('wnsimple_hypernym_transitive', 'hypernym')
            isa_parts_v3['predicate'] = isa_parts_v3['predicate'].str.replace('wnsimple_parts_transitive', 'meronym')
            isa_parts_v3['predicate'] = isa_parts_v3['predicate'].str.replace('wnsimple_parts_direct_hypernyms', 'meronym')
            isa_parts_v3['predicate'] = isa_parts_v3['predicate'].str.replace('wnsimple_parts_direct', 'meronym')

            # filtering inaccurate / unclear data:
            isa_parts_v3 = isa_parts_v3[~isa_parts_v3['object'].isin(['section', 'artifact','bridge','s','l','m','beard','moustache'])]

            # adding generally true for all matter
            if False:
                atoms = isa_parts_v3[isa_parts_v3['predicate'] == 'meronym'].copy(deep=True)
                atoms['object'] = 'atom'
                atoms = atoms.drop_duplicates()
                atoms = atoms[~atoms['subject'].isin(['sang','elastic','inn','hour','ton','bluff','daylight','education','carnation'])]

                molecule = isa_parts_v3[isa_parts_v3['predicate'] == 'meronym'].copy(deep=True)
                molecule['object'] = 'molecule'
                molecule = molecule.drop_duplicates()
                molecule = molecule[~molecule['subject'].isin(['sang', 'elastic', 'inn', 'hour', 'ton', 'bluff', 'daylight','education','carnation'])]

                isa_parts_v3 = isa_parts_v3.append(atoms, ignore_index=True)
                isa_parts_v3 = isa_parts_v3.append(molecule, ignore_index=True)
                isa_parts_v3 = isa_parts_v3.append([{'subject': 'atom','predicate': 'meronym', 'object': 'proton'},
                                                    {'subject': 'atom', 'predicate': 'meronym', 'object': 'neutron'},
                                                    {'subject': 'atom', 'predicate': 'meronym', 'object': 'electron'},
                                                    {'subject': 'atom', 'predicate': 'meronym', 'object': 'quark'},
                                                    {'subject': 'molecule', 'predicate': 'meronym', 'object': 'atom'},
                                                    {'subject': 'molecule', 'predicate': 'meronym', 'object': 'neutron'},
                                                    {'subject': 'molecule', 'predicate': 'meronym', 'object': 'proton'}])


            for id, edge in isa_parts_v3.iterrows():
                edge_dict = edge.to_dict()
                edge_dict.update({'validity':'always true', 'source': 'wordnet/conceptnet'})
                self.kb.append(edge_dict)

        def add_conceptnet_part_of(self):
            conceptnet_meronyms = self._ConceptNet_full[(self._ConceptNet_full['relation'] == '/r/PartOf') & \
                                                        (self._ConceptNet_full['weights'] > 1)]

            for ind, r in conceptnet_meronyms.iterrows():
                # logger.error(p['relation'])
                self.kb.append({'subject': r['end_term'],
                                'predicate': 'part of',
                                'source': 'conceptnet',
                                'object': r['start_term'],
                                'validity': 'always true'})

        def add_func_rel_from_wikidata(self):
            if not os.path.exists('data/resources/spouses.csv'):
                data = WikiData().get_spouses()
                data.to_csv('data/resources/spouses.csv', index=False)
            else:
                data = pd.read_csv('data/resources/spouses.csv')
            data = data.drop_duplicates(subset=['person', 'spouse'])
            same_family_name = data['person'].str.split(' ').str[-1] == data['spouse'].str.split(' ').str[-1]
            data.loc[same_family_name, 'spouse'] = data.loc[same_family_name, 'spouse'].str.split(' ').str[0:-1].apply(
                lambda x: ' '.join(x))
            self.convert_func_rel_data_to_triplets(data, 'spouse', 'is the spouse of', 'person', 4, 'spouse', sample_size=2000)

            if not os.path.exists('data/resources/person_children.csv'):
                data = WikiData().get_children()
                data.to_csv('data/resources/person_children.csv', index=False)
            else:
                data = pd.read_csv('data/resources/person_children.csv')
            data = data.drop_duplicates(subset=['person', 'child'])
            same_family_name = data['person'].str.split(' ').str[-1] == data['child'].str.split(' ').str[-1]
            data.loc[same_family_name, 'child'] = data.loc[same_family_name, 'child'].str.split(' ').str[0:-1].apply(lambda x: ' '.join(x))
            self.convert_func_rel_data_to_triplets(data, 'child', 'is the child of', 'person', 4, 'child', sample_size=2000)

            data = pd.read_excel('data/resources/fortune1000.xlsx')
            # A few companies have more than one CEO, lets discarde them for now
            data.rename(columns={'title': 'company'}, inplace=True)
            data.rename(columns={'City': 'headquarters'}, inplace=True)
            # data = data.explode('CEO')
            data = data[data['CEO'].str.split('/').apply(len) == 1]
            self.convert_func_rel_data_to_triplets(data, 'CEO', 'is the CEO of', 'company', 1, 'CEO')
            self.convert_func_rel_data_to_triplets(data, 'headquarters', 'is the headquarters of', 'company', 1, 'headquarter')

            if not os.path.exists('data/resources/company_founders.csv'):
                data = WikiData().get_company_founders(2000)
                data.to_csv('data/resources/company_founders.csv', index=False)
            else:
                data = pd.read_csv('data/resources/company_founders.csv')
            data = data.drop_duplicates(subset=['founder', 'company'])
            self.convert_func_rel_data_to_triplets(data, 'founder', 'founded', 'company', 4, 'founder')

            data = pd.read_csv('data/resources/netflix_titles.csv')
            data = data[data['type'] == 'Movie']
            data.rename(columns={'title': 'movie'}, inplace=True)
            # A few companies have more than one CEO, lets discarde them for now
            data.rename(columns={'release_year': 'release year'}, inplace=True)
            self.convert_func_rel_data_to_triplets(data, 'release year', 'is the release year of', 'movie', 1, 'release year', sample_size=1000)

            data = data[data['director'].notnull()]
            data['director'] = data['director'].str.split(', ')
            data = data.explode('director')
            self.convert_func_rel_data_to_triplets(data, 'director', 'is director of', 'movie', 3, 'director')

            capitals = pd.read_csv('data/resources/cities.csv')
            data = capitals[capitals['city'] == capitals['capital']]
            negative_data = capitals[capitals['city'] != capitals['capital']]
            negative_data = negative_data.sample(n=1000, random_state=17)
            only_capitals = capitals[['country', 'capital']].drop_duplicates()
            only_capitals.columns = ['country', 'city']
            only_capitals = only_capitals.groupby('country').apply(lambda x: list(x['city'])).to_dict()
            self.convert_func_rel_data_to_triplets(data, 'city', 'is the capital of', 'country', 1, 'capital', negative_data, only_capitals)

            data = pd.read_csv('data/resources/bands.csv')
            # getting rid of IDs
            data = data[~data['musician'].str.startswith('Q')]
            # lets get rid of the non ascii names...
            data = data[data['musician'].apply(lambda s: len(s) == len(s.encode()))]
            data = data[data['band'].apply(lambda s: len(s) == len(s.encode()))]
            self.convert_func_rel_data_to_triplets(data, 'musician', 'is a member of', 'band', 4, 'band member')

            data = pd.read_csv('data/resources/superbowl.csv')[['Date', 'Winner', 'Loser']]
            data.columns = ['year', 'winner', 'loser']
            data['year'] = data['year'].str.split(' ').str[-1]
            self.convert_func_rel_data_to_triplets(data, 'winner', 'won the Super Bowl of', 'year', 1, 'super bowl winner')
            self.convert_func_rel_data_to_triplets(data, 'loser', 'lost the Super Bowl of', 'year', 1, 'super bowl loser')

        def convert_func_rel_data_to_triplets(self, data, subject, predicate, object, max_count, func_rel, \
                                  negative_data=None, object_index=None, sample_size=None):
            if object_index is None:
                object_index = data.groupby(object).apply(lambda x: list(x[subject])).to_dict()

            data['count'] = data[object].map({k: len(v) for k, v in object_index.items()})
            data = data[data['count'] <= max_count]
            # we would like the count of instance to be evenly distributed for each func_rel.
            if sample_size is None:
                data = uniform_sample_by_column(data, 'count', object, int(data['count'].value_counts().mean()))
            else:
                data = uniform_sample_by_column(data, 'count', object, sample_size)

            # We sample negative example by randomizing the value of the subject, and making sure the result does not
            # exist in the positive examples.
            if negative_data is None:
                # over_sample negative examples - these are solved using the counting rule, thus we would like to over represent them.
                negative_data = data.copy(deep=True).sample(frac=1.5, replace=True, random_state=17)
                negative_data.loc[:, subject] = list(negative_data.loc[:, subject].sample(frac=1, random_state=0))
                negative_data = pd.merge(negative_data, data, on=[subject, object, 'count'], how="outer", indicator=True)
                negative_data = negative_data[negative_data['_merge'] == 'left_only']
                if len(negative_data) > len(data) * 3:
                    negative_data = negative_data.sample(n=(len(data) * 3), random_state=17)
            else:
                negative_data['count'] = negative_data[object].map({k: len(v) for k, v in object_index.items()})

            #for type, instances in zip(['positive', 'negative'], [data, negative_data]):
            #    for i, e in instances.iterrows():
            #        self.kb.append({'subject': e[subject],
            #                        'predicate': func_rel,
            #                        'object': e[object],
            #                        'source': 'wikidata',
            #                        'validity': 'always true' if type == 'positive' else 'never true'})
            for i, e in data.iterrows():
                self.kb.append({'subject': e[subject],
                                'predicate': func_rel,
                                'object': e[object],
                                'source': 'wikidata',
                                'validity': 'always true'})

            # adding the hyperyms of each function relation object
            for i, e in data.drop_duplicates(subset=[object]).iterrows():
                self.kb.append({'subject': e[object],
                                'predicate': 'hypernym',
                                'object': object,
                                'source': 'wikidata',
                                'validity': 'always true'})

        def add_always_true_conceptnet_properties(self):
            cn_edges_filt = self._ConceptNet_full
            cn_edges_filt = cn_edges_filt[
                (cn_edges_filt['relation'] != '/r/HasProperty') & \
                (cn_edges_filt['relation'] != '/r/ReceivesAction') & \
                (cn_edges_filt['relation'] != '/r/AtLocation') & \
                (cn_edges_filt['relation'] != '/r/UsedFor') & \
                (cn_edges_filt['relation'] != '/r/DerivedFrom') & \
                (cn_edges_filt['relation'] != '/r/DistinctFrom') & \
                (cn_edges_filt['relation'] != '/r/NotDesires') & \
                (cn_edges_filt['relation'] != '/r/RelatedTo') & \
                (cn_edges_filt['relation'] != '/r/FormOf') & \
                (cn_edges_filt['relation'] != '/r/Synonym') & \
                (cn_edges_filt['relation'] != '/r/SimilarTo') & \
                (cn_edges_filt['relation'] != '/r/MannerOf') & \
                (cn_edges_filt['relation'] != '/r/HasContext') & \
                (cn_edges_filt['relation'] != '/r/EtymologicallyRelatedTo') & \
                ~(cn_edges_filt['relation'].str.contains('/r/dbpedia'))]
            cn_edges_filt = cn_edges_filt.sort_values(by='weights').drop_duplicates(subset=['start_term', 'relation', 'end_term'],
                                                                                    keep='last')
            # TODO move this to manual KB.
            end_term_filter = ['information', 'equipment', 'part', 'applicant', 'cloud thinking', 'state of matter', 'shape', 'drug', \
                               'plane figure', 'person', 'pets', 'food', 'assistant', 'make music', 'content', 'punishment', 'gathering', \
                               'wind instrument', 'unit', 'another name for homosexual', 'consequence', 'time period']
            cn_edges_filt = cn_edges_filt[~cn_edges_filt['end_term'].isin(end_term_filter)]

            selected_hypernyms  = {e['object'] for e in self.sample({'predicate':'hypernym'})}
            hypernym_relation_filter = {'/r/CapableOf': ['machine', 'bird', 'drug', 'machine', 'person']}
            pbar1 = tqdm(desc='Rules linked to hypernym objects')
            for hp in selected_hypernyms:
                for ind, r in cn_edges_filt[(cn_edges_filt['start_term'] == hp) & \
                                                    (cn_edges_filt['weights'] > 1)].iterrows():
                    if r['relation'] in self.relation_templates_statement and \
                            not (r['relation'] in hypernym_relation_filter and hp in hypernym_relation_filter[r['relation']]):
                        # logger.error(p['relation'])
                        self.kb.append({'subject': r['start_term'],
                                        'predicate': r['relation'],
                                        'source': 'conceptnet',
                                        'object': r['end_term'],
                                        'validity': 'always true'})
                        pbar1.update(1)
            pbar1.close()

        def add_never_true(self):
            # Except for Desires and NotDesires we would like to have negative examples for capable of IsA, PartOf, and CapableOf
            # not the antonyms and distinct from are already negative of IsA, but the numbers are much less.
            cn_negative = self._ConceptNet_full[self._ConceptNet_full['relation'].isin(['/r/IsA', \
                                                                '/r/PartOf', '/r/CapableOf'])].copy(deep=True)
            self.negative_class_relations = ['/r/IsA', '/r/PartOf', '/r/CapableOf']

            # just randomizing the start term.
            for relation in self.negative_class_relations:
                cn_negative.loc[cn_negative['relation'] == relation, 'start_term'] = \
                    list(cn_negative.loc[cn_negative['relation'] == relation, 'start_term'].sample(frac=1, random_state=0))

            # removing true rules
            cn_negative = cn_negative.merge(self._ConceptNet_full, on=['start_term', 'relation','end_term'], how='left')
            cn_negative = cn_negative[cn_negative['start_y'].isnull()]

            pbar2 = tqdm(desc='Never true added')
            for ind, r in tqdm(cn_negative.iterrows(), desc='never_true'):
                if r['relation'] in self.relation_templates_statement:
                    # logger.error(p['relation'])
                    self.kb.append({'subject': r['start_term'],
                                    'predicate': r['relation'],
                                    'object': r['end_term'],
                                    'source': 'conceptnet',
                                    'validity': 'never true'})
                    pbar2.update(1)
            pbar2.close()

        def add_situational_possetion(self):
            pass
            # Imaginary counting
            # Has 3 dogs, (each has a name)
            # Has 2 cars (car types)
            # 4 televisions
            # 5 fruits (fruit names)
            # 2 cars

            # imaginary occupasions - John is the mayor
            # the town has 2 mayers...
            #examples.append({'subject': e[subject], 'predicate': predicate, 'object': e[object]}

        def lookup(self, triplet):
            triplet_tuple = (triplet['subject'], triplet['predicate'], triplet['object'])
            if triplet_tuple in self.kb_full_index:
                return self.kb_full_index.loc[triplet_tuple].reset_index().to_dict(orient='rows')
            else:
                return []

        def sample(self, query, tar_tag=None, sample_limit=None, select=None):
            logger.debug(f'sample {query} for target {tar_tag}')
            output = []

            # supporting non list matching
            for field,val in query.items():
                if type(val) != list:
                    query[field] = [val]

            shuffled_kb = self.kb
            # TODO we still don't have an index for our KB because it is simple and small, in the future we will add one...
            if sample_limit is not None:
                sample_limit_count = {}
                field, limit = sample_limit
                shuffled_kb = random.sample(self.kb, len(self.kb))

            # TODO temporary, we need to index all the data:
            if set(query.keys()) == {'subject','predicate'} and len(query['predicate']) == 1 and len(query['subject']) == 1:
                if (query['subject'][0], query['predicate'][0]) in self.subject_predicate_index.index:
                    return self.subject_predicate_index.loc[(query['subject'][0], query['predicate'][0] )].reset_index().to_dict(orient='rows')

            if set(query.keys()) ==  {'predicate','object'} and len(query['predicate']) == 1 and len(query['object']) == 1:
                if (query['predicate'][0], query['object'][0]) in self.predicate_object_index.index:
                    return self.predicate_object_index.loc[(query['predicate'][0],query['object'][0])].reset_index().to_dict(orient='rows')

            for edge in shuffled_kb:
                match = True
                for key, val in query.items():
                    if '_not_in' in key:
                        if edge[key.replace('_not_in','')] in val:
                            match = False
                    else:
                        if edge[key] not in val:
                            match = False

                if sample_limit is not None:
                    if edge[field] not in sample_limit_count:
                        sample_limit_count[edge[field]] = 1
                    else:
                        sample_limit_count[edge[field]] += 1
                        if sample_limit_count[edge[field]] > limit:
                            continue

                if select is not None:
                    edge = {k:v for k, v in copy.deepcopy(edge).items() if k in select}
                if match:
                    if tar_tag is not None:
                        output.append({tar_tag: edge})
                    else:
                        output.append(edge)

            #if sample_limit is not None:
            #    print(pd.DataFrame(sample_limit_count).sort_values())

            return output

        def to_pseudo_language(self, triplet, is_rule,
                               reverse_validity=False,
                               use_shorthand=False,
                               use_hypothetical_statement=False,
                               nlg_sampling=False):
            subject = triplet['subject']
            predicate = triplet['predicate']
            object = triplet['object']
            template = None
            if is_rule:
                if reverse_validity:
                    template = self.negative_rules_templates[predicate]
                else:
                    if use_shorthand:
                        template = self.rules_templates_shorthand[predicate]
                    else:
                        template = self.rules_templates[predicate]
            else:
                if reverse_validity:
                    if use_shorthand:
                        template = self.relation_templates_negative_statement_shorthand[predicate]
                    else:
                        template = self.relation_templates_negative_statement[predicate]
                else:
                    if use_shorthand:
                        template = self.relation_templates_statement_shorthand[predicate]
                    else:
                        template =  self.relation_templates_statement[predicate]

            if not isinstance(template, str):
                template = random.choice(template) if nlg_sampling else template[0]
            output = template.replace('[subject]', subject).replace('[object]', object)
            if use_hypothetical_statement:
                output = 'Given only the context, it’s possible that ' + output
            return output

        #@profile
        def connect(self, connect_to, constraints, src_tags, connection_point, tar_tag , max_to_connect=None):
            # connection_point = [{src_field1:tar_field1, (and) src_field2:tar_field2}, or {src_field3:tar_field3 (and) ... }] OR
            # connection_point_subset_max_sample = for each connection point subset, do sample 0 to K.
            logger.info(f'connect. {tar_tag} --> {src_tags} ')

            # sampling candidate edges, and moving to DF for efficiency in selecting.
            candidate_edges = pd.DataFrame(self.sample(constraints))

            # ADDING INDEXES?
            connection_point_indexes = {}
            for or_term in connection_point:
                index_name = str(list(or_term.values()))
                if index_name not in connection_point_indexes:
                    connection_point_indexes[index_name] = candidate_edges.reset_index().set_index(list(or_term.values()))

            # Link the connection to existing tags.
            output = []
            for edge in connect_to:
                for src_tag in src_tags:
                    all_cand_inds = []
                    cand_inds = {}
                    for term_num, or_term in enumerate(connection_point):
                        index_name = str(list(or_term.values()))
                        src_val = tuple([edge[src_tag][k] for k,v in or_term.items()]) if len(or_term.keys()) > 1 else edge[src_tag][
                            next(iter(or_term))]
                        if src_val in connection_point_indexes[index_name].index:
                            cand_inds[term_num] = set(connection_point_indexes[index_name].loc[[src_val],'index'])
                        else:
                            cand_inds[term_num] = {}
                        all_cand_inds += cand_inds[term_num]

                    if max_to_connect is not None and len(all_cand_inds) > max_to_connect:
                        all_cand_inds = random.sample(all_cand_inds, max_to_connect)

                    for cand_ind in all_cand_inds:
                        new_edge = copy.deepcopy(edge)
                        new_edge.update({tar_tag: candidate_edges.loc[cand_ind].to_dict()})
                        output.append(new_edge)

            return output

        def connect_downward_monotone(self, connect_to, scope, property, tar_tag):
            # for each example, there is one downword monotone.
            output = []
            for edge in connect_to:
                new_edge = copy.deepcopy(edge)
                new_edge.update({tar_tag: {'subject': edge[scope]['subject'],
                                           'predicate': edge[property]['predicate'],
                                           'object': edge[property]['object'],
                                           'validity': edge[property]['validity']}})
                output.append(new_edge)
            return output

        def sample_any_subset(self):
            pass



    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if TeachAIKB.__instance is None:
            # Create and remember instance
            TeachAIKB.__instance = TeachAIKB.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Singleton__instance'] = TeachAIKB.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)










