import random
import _jsonnet, json
import logging
import hashlib
import os 
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import math
from LeapOfThought.resources.teachai_kb import TeachAIKB
from LeapOfThought.common.general import num2words1, bc
from LeapOfThought.common.data_utils import uniform_sample_by_column, pandas_multi_column_agg

# This is mainly for testing and debugging  ...
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 200)
pd.set_option("display.colheader_justify","left")
import numpy as np
from LeapOfThought.common.file_utils import upload_jsonl_to_s3, save_jsonl_to_local, is_path_creatable

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ArtiSet():
    def __init__(self, args):

        random.seed(17)
        np.random.seed(1234)
        self._np_seed = np.random.RandomState(17)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path) ,'r') as f:
            self._config = json.load(f)[self.artiset_name]

        if args.__contains__('variant') and len(args.variant) > 0:
            self._output_file = args.output_file.replace('.jsonl','_' + args.variant + '.jsonl')
            if len(args.experiment_version) > 0:
                self._output_file = self._output_file.replace('.jsonl', '_' + args.experiment_version + '.jsonl')
        else:
            self._output_file = args.output_file

        self._split = args.split_by_field
        self._incorrect_beliefs = None
        if "incorrect_beliefs_file" in args and args.incorrect_beliefs_file:
            with open(args.incorrect_beliefs_file, 'r') as file:
                self._incorrect_beliefs = [json.loads(line.strip()) for line in file]
        self._save_sample = args.save_sample
        self.artiset_data = []

    def append_teachyourai_format_example(self, example, do_print=False, append_to_list=None):
        """append_teachyourai_format_example() is method implemented in ArtiSet class and takes an example dict
        (that must contain a "phrase", "answer") and converts it to a BooleanQA format

        Args:
            example (dict): an example containing question,answer,dist1,dist2 fields
            do_print (bool): just for debuging
            num_choices (int): number of choices in question (between 2 and 5)
            append_to_list (list): a

        Returns:

        """

        if 'context' not in example:
            example['context'] = ''

        if 'id' not in example:
            example['id'] = self.create_qid(example)

        if do_print:
            print('a:%s d1:%s d2:%s || Q:%s' % (example['phrase'], example['answer']))

        if append_to_list is not None:
            append_to_list.append(example)
        else:
            self.artiset_data.append(example)

    @staticmethod
    def create_qid(example):
        m = hashlib.md5()
        m.update(example['phrase'].encode())
        m.update(example['context'].encode())
        # boolean examples have binary answer (int 0 or 1)
        m.update(str(example['answer']).encode())
        return m.hexdigest()

    def split_by_columns(self):
        split_columns = self._split.split(',')
        examples = self.examples_meta
        indexes = {}

        # check the split columns are in the data
        if len(set(split_columns) - set(examples.columns)) != 0:
            raise (ValueError("split columns used to split dev/test and train set do not exist the examples_meta!"))

        all_objs = []
        for split_column in split_columns:
            all_objs += list(examples[split_column])

        #best_train_inds, best_dev_inds, best_test_inds = [], [], []
        inds = [i for i in range(len(self.artiset_data))]
        random.seed(17)
        random.shuffle(inds)

        if len(split_columns) > 1:
            test_inds = random.sample(inds, self._config['test_dev_size'][0])
            inds = list(set(inds) - set(test_inds))
            dev_inds = random.sample(inds, self._config['test_dev_size'][1])
            dev_test_examples = examples.iloc[test_inds + dev_inds]

            dev_test_objs = []
            for split_column in split_columns:
                dev_test_objs += list(dev_test_examples[split_column])
            dev_test_objs = pd.Series(list(set(dev_test_objs)))
        else:
            # We'll choice the test-dev examples from values of split that have the lowest number of examples.
            # this will insure we are choosing to highest amount of training examples that are still disjoint on split_columns[0] from dev+test
            split_columns_value_counts = examples[split_columns[0]].value_counts().sort_values().cumsum().reset_index()
            start_ind = split_columns_value_counts[split_columns_value_counts[split_columns[0]] > \
                                                 sum(self._config['test_dev_size'])].index[0] + 1
            dev_test_objs = list(split_columns_value_counts['index'][0:start_ind])
            dev_test_examples = examples[examples[split_columns[0]].isin(dev_test_objs)]
            inds = list(dev_test_examples.index)
            test_inds = random.sample(inds, self._config['test_dev_size'][0])
            inds = list(set(inds) - set(test_inds))
            dev_inds = random.sample(inds, self._config['test_dev_size'][1])

        for split_column in split_columns:
            indexes[split_column] = examples.set_index(split_column)

        dev_ids = set()
        not_in_train_ids = set()
        for split_column in split_columns:
            dev_ids = dev_ids & set(indexes[split_column][indexes[split_column].index.isin(dev_test_objs)]['qid'])
            not_in_train_ids = not_in_train_ids | set(indexes[split_column][indexes[split_column].index.isin(dev_test_objs)]['qid'])

        train_examples = examples.loc[~examples['qid'].isin(not_in_train_ids), :]
        train_inds = list(train_examples.index)

        if len(train_inds) > self._config['max_number_of_examples']:
            train_inds = train_inds[0:self._config['max_number_of_examples']]
        random.shuffle(train_inds)

        print("total dev-test examples available: %d" % (len(dev_test_examples)))
        print("split produced %d training examples" % (len(train_inds)))
        return train_inds, dev_inds, test_inds

    def save_dataset(self):
        """save_dataset() automatically saves the artiset
        if the config output_file contains the string _sample.jsonl it will be saved in a more readable format
        otherwise it will split the examples in self.artiset_data into train, dev, test and save them in s3
        if output_file startswith s3:// otherwise locally. (If output_file is empty, it will not save)

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        """

        # Move non-required columns to metadata:
        artiset_data_with_metadata = []
        for example in self.artiset_data:
            if 'metadata' not in example:
                new_example = {'metadata':{}}
            else:
                new_example = {'metadata': example['metadata']}
            new_example.update({k:example[k] for k in ['id', 'phrase', 'context', 'answer']})
            new_example['metadata'].update({k: example[k] for k in set(example.keys()) - {'id', 'phrase', 'context', 'answer','metadata'}})
            artiset_data_with_metadata.append(new_example)
        self.artiset_data = artiset_data_with_metadata

        # splitting
        if len(self._split) > 0:
            train_inds, dev_inds, test_inds = self.split_by_columns()
        elif 'split' in self.examples_meta:
            test_inds = list(self.examples_meta[self.examples_meta['split'] == 'test'].index)
            dev_inds = list(self.examples_meta[self.examples_meta['split'] == 'dev'].index)
            train_inds = list(self.examples_meta[self.examples_meta['split'] == 'train'].index)
            random.seed(17)
            random.shuffle(train_inds)
            #random.shuffle(test_inds)
            #random.shuffle(dev_inds)
            test_inds = test_inds[0: self._config['test_dev_size'][0]]
            dev_inds = dev_inds[0:self._config['test_dev_size'][1]]
            train_inds = train_inds[0:self._config['max_number_of_examples']]
        else:
            inds = [i for i in range(len(self.artiset_data))]
            random.seed(17)
            random.shuffle(inds)
            test_inds = inds[0:self._config['test_dev_size'][0]]
            dev_inds = inds[self._config['test_dev_size'][0]:sum(self._config['test_dev_size'])]
            train_inds = inds[sum(self._config['test_dev_size']):]


        if self._output_file.startswith('s3://'):
            save_func = upload_jsonl_to_s3
        elif is_path_creatable(self._output_file) and len(self._output_file) > 0:
            save_func = save_jsonl_to_local
        else:
            # Do nothing
            return

        if self._save_sample:
            if 'split' in self.examples_meta.columns:
                logger.info(f"size of each split:\n{self.examples_meta['split'].value_counts()}")
            random.seed(17)
            if len(self.artiset_data) > 100:
                self.artiset_data = random.sample(self.artiset_data,100)
            save_func(self._output_file, self.artiset_data, sample_indent=self._save_sample)
        else:
            logger.info('uploading %d,%d,%d test,dev,train examples' % (len(test_inds),len(dev_inds),len(train_inds)))
            save_func(self._output_file.replace('.jsonl', '_test.jsonl'), [self.artiset_data[i] for i in test_inds])
            save_func(self._output_file.replace('.jsonl', '_dev.jsonl'), [self.artiset_data[i] for i in dev_inds])
            save_func(self._output_file.replace('.jsonl', '_train.jsonl'), [self.artiset_data[i] for i in train_inds])
            if len(self.examples_meta) > 0:
                save_func(self._output_file.replace('.jsonl', '_meta.jsonl'), self.examples_meta.to_dict(orient='rows'))

        return train_inds, dev_inds, test_inds

    def save_single_split(self, split_data, split):
        inds = [i for i in range(len(split_data))]
        random.seed(17)
        random.shuffle(inds)

        if self._output_file.startswith('s3://'):
            save_func = upload_jsonl_to_s3
        elif is_path_creatable(self._output_file) and len(self._output_file) > 0:
            save_func = save_jsonl_to_local
        else:
            # Do nothing
            return

        si = self._output_file.find('_sample') > -1
        save_func(self._output_file.replace('.jsonl', '_' + split + '.jsonl'), [split_data[i] for i in inds], sample_indent=si)

    def save_aux_data(self, output_file, data):
        if output_file.startswith('s3://'):
            save_func = upload_jsonl_to_s3
        elif is_path_creatable(output_file) and len(output_file) > 0:
            save_func = save_jsonl_to_local
        else:
            # Do nothing
            return

        si = output_file.find('_sample') > -1
        save_func(output_file, data, sample_indent=si)

    def build_artificial_dataset(self,args):
        pass

    def resplit(self, args):
        logger.error('Not implemented for this artiset')

    def build_statement_rule_property_examples(self, examples, split, statement_tag='statement', ablate_same_distractor_fields = 1.0,\
                rule_tags=['implicit_rule','property'], distractor_tags = ['distractors'], ablation_list=[], use_shorthand=False, \
                                               nlg_sampling=False, reverse_validity_frac=0):


        # computing ID before ablations on the statement and rule tags:
        for i, example in enumerate(examples):
            m = hashlib.md5()
            # note that the tags for ID creation are always the same!
            for tag in [statement_tag] + rule_tags:
                if tag in example:
                    if type(example[tag]) == list:
                        for e in example[tag]:
                            m.update(e['subject'].encode())
                            m.update(e['predicate'].encode())
                            m.update(e['object'].encode())
                            m.update(e['validity'].encode())
                    else:
                        m.update(example[tag]['subject'].encode())
                        m.update(example[tag]['predicate'].encode())
                        m.update(example[tag]['object'].encode())
                        m.update(example[tag]['validity'].encode())
            example['id'] =  m.hexdigest()

        # Ablations
        # now that all the examples are ready, we can ablate as needed:
        random.seed(17)
        for ablation in ablation_list:
            if len(ablation) == 3:
                fields, fraction, condition = ablation
                examples_cands = [e for e in examples if e[condition[0]] in condition[1]]
            else:
                fields, fraction = ablation
                examples_cands = examples
            example_to_ablate = random.sample(examples_cands, int(fraction * float(len(examples))))
            for e in example_to_ablate:
                for field in fields:
                    if field in e:
                        del e[field]
                    # for every field we ablate we must ablate the same field from distractors!
                    if random.random() < ablate_same_distractor_fields:
                        for distractor_tag in distractor_tags:
                            if distractor_tag in e:
                                if field in e[distractor_tag]:
                                    del e[distractor_tag][field]

        random.seed(17)
        for i, example in enumerate(examples):
            context_rules = []
            # adding actual rules
            for rule_tag in rule_tags:
                if rule_tag in example:
                    rules = example[rule_tag]
                    if not type(rules) == list:
                        rules = [rules]
                    for rule in rules:
                        reverse_validity = not rule['validity'] == 'always true'
                        context_rules.append(TeachAIKB().to_pseudo_language(rule,
                                                                            is_rule=True, reverse_validity=reverse_validity,
                                                                            use_shorthand=use_shorthand, nlg_sampling=nlg_sampling))
            # adding distractors
            for rule_tag in distractor_tags:
                if rule_tag in example:
                    for field, tag_distractors in example[rule_tag].items():
                        for rule in tag_distractors:
                            rule_list = rule
                            if not type(rule_list) == list:
                                rule_list = [rule_list]
                            for r in rule_list:
                                reverse_validity = not r['validity'] == 'always true'
                                context_rules.append(TeachAIKB().to_pseudo_language(r, is_rule=True, reverse_validity=reverse_validity,
                                                                                    use_shorthand=use_shorthand,
                                                                                    nlg_sampling=nlg_sampling))

            use_hypothetical_statement = False
            if 'is_hypothetical_statement' in example and example['is_hypothetical_statement']:
                use_hypothetical_statement = True

            answer = 1 if example[statement_tag]['validity'] == 'always true' else 0

            if self.variant != 'statement_subject_lang_selectivity':

                if random.random() < reverse_validity_frac:
                    answer = 1 - answer
                    reverse_validity = True
                else:
                    reverse_validity = False
                phrase = TeachAIKB().to_pseudo_language(example[statement_tag], is_rule=False, use_shorthand=use_shorthand,
                                                        use_hypothetical_statement=use_hypothetical_statement,
                                                        nlg_sampling=nlg_sampling, reverse_validity=reverse_validity)
            else:
                statement_dict = deepcopy(example[statement_tag])
                statement_dict['subject'] = random.sample(['foo','blah','ya','qux','aranglopa','foltopia','cakophon','baz','garply'], 1)[0]
                phrase = TeachAIKB().to_pseudo_language(statement_dict, is_rule=False, use_shorthand=use_shorthand,
                                                        use_hypothetical_statement=use_hypothetical_statement,
                                                        nlg_sampling=nlg_sampling)

            # creating a unique set of rules that does not include the statement.
            context_rules = list(set(context_rules))
            # set order is random!! so we need to fix the order the get a replicable order.
            context_rules = sorted(context_rules)
            random.shuffle(context_rules)

            example.update({'phrase': phrase, \
                            'answer': answer,
                            'context': ' '.join(context_rules),
                            'split': split,
                            'rules': context_rules})

            # append_teachyourai_format_example() is method implemented in ArtiSet class and takes an example dict
            # (that must contain a "phrase", "answer") and converts it to a BooleanQA format
            self.append_teachyourai_format_example(example, do_print=False)
            self.examples_meta.append(deepcopy(example))

    def print_examples(self, sample):
        random.seed(7)
        example_inds = random.sample(range(len(self.artiset_data)), sample)
        ## Printing a sample!
        for ind in example_inds:
            example = self.artiset_data[ind]
            if 'statement' in example:
                statement = example['statement']
                rules = '\n'.join(example['rules'])
                e = f"{example['id']}({example['split']}):\n{bc.BOLD}Q:{bc.ENDC}{example['phrase']} {bc.BOLD}A:{bc.ENDC}{example['answer']}\n{bc.BOLD}C:{bc.ENDC}{rules} "
                e = e.replace(statement['object'], f"{bc.Blue}{statement['object']}{bc.ENDC}")
                e = e.replace(statement['predicate'], f"{bc.Green}{statement['predicate']}{bc.ENDC}")
                e = e.replace(str(statement['subject']), f"{bc.Magenta}{statement['subject']}{bc.ENDC}")
                if 'hypernym' in example:
                    hypernym = example['hypernym']['object']
                    e = e.replace(str(hypernym), f"{bc.Cyan}{hypernym}{bc.ENDC}")
                e = e.replace('not', f"{bc.Red}not{bc.ENDC}")
                e = e.replace('type', f"{bc.Yellow}type{bc.ENDC}")
                if 'num_of_instances' in example:
                    e = e.replace(' ' + num2words1[example['num_of_instances']].lower() + ' ' \
                                  , f"{bc.Red} {num2words1[example['num_of_instances']].lower()} {bc.ENDC}")
                for number in 'one', 'two', 'three', 'four', 'five':
                    e = e.replace(' ' + number + ' ', f"{bc.Cyan} {number} {bc.ENDC}")
            else:
                e = f"{example['id']}({example['split']}):\n{bc.BOLD}Q:{bc.ENDC}{example['phrase']} {bc.BOLD}A:{bc.ENDC}{example['answer']}\n{bc.BOLD}C:{bc.ENDC}{example['context']} "
            print(e + '\n')

    def create_subject_filter_lookup(self, examples, sample_on=None, avoid_mixing=None):
        if sample_on is not None:
            triplets_to_sample_on = [e[sample_on] for e in examples]
        else:
            triplets_to_sample_on = examples

        # building subject filter lookup:
        subject_filter_lookup = {}
        rules_to_sample_df = pd.DataFrame(triplets_to_sample_on)
        for curr_subject, matching_records in tqdm(rules_to_sample_df.groupby('subject')):
            subject_to_filter = {curr_subject}

            if avoid_mixing is not None and 'predicates' in avoid_mixing:
                subject_to_filter |= set(
                    rules_to_sample_df[~rules_to_sample_df['predicate'].isin(set(matching_records['predicate']))]['subject'])

            if avoid_mixing is not None and 'hyponyms' in avoid_mixing:
                subject_to_filter |= {e['subject'] for e in TeachAIKB().sample({'predicate': 'hypernym', 'object': curr_subject})}

            if avoid_mixing is not None and 'co-hyponyms' in avoid_mixing:
                subject_is_hyponym_of = {e['object'] for e in TeachAIKB().sample({'subject': curr_subject, 'predicate': 'hypernym'})}
                subject_to_filter |= {e['subject'] for e in
                                      TeachAIKB().sample({'predicate': 'hypernym', 'object': list(subject_is_hyponym_of)})}

            if avoid_mixing is not None and 'co-meronyms' in avoid_mixing:
                subject_is_meronym_of = {e['subject'] for e in TeachAIKB().sample({'predicate': 'meronym', 'object': curr_subject})}
                subject_to_filter |= {e['object'] for e in
                                      TeachAIKB().sample({'predicate': 'meronym', 'subject': list(subject_is_meronym_of)})}
            subject_filter_lookup[curr_subject] = subject_to_filter

        return subject_filter_lookup

    #@profile
    def self_negative_subject_sample(self, examples, sample_on = None, avoid_mixing=None, over_sample = 1.0):
        examples = deepcopy(examples)
        if sample_on is not None:
            triplets_to_sample_on = [e[sample_on] for e in examples]
        else:
            triplets_to_sample_on = examples

        subject_filter_lookup = self.create_subject_filter_lookup(examples, sample_on, avoid_mixing)
        output = []
        examples_to_gen_from = deepcopy(examples) + random.sample(deepcopy(examples),int((over_sample - 1) * len(examples)))
        for i,example in tqdm(enumerate(examples_to_gen_from)):
            # sometimes we just want a list of triplets, with no specific dictionary field called "sample_on" ...
            if sample_on is not None:
                curr_triplet = example[sample_on]
            else:
                curr_triplet = example
            curr_subject = curr_triplet['subject']

            if sample_on is not None:
                new_edge = deepcopy(
                    random.sample([e for e in examples if e[sample_on]['subject'] not in subject_filter_lookup[curr_subject]], 1)[0])
                new_edge[sample_on]['predicate'] = deepcopy(curr_triplet['predicate'])
                new_edge[sample_on]['object'] = deepcopy(curr_triplet['object'])
                new_edge[sample_on]['validity'] = 'never true'
            else:
                new_edge = deepcopy(
                    random.sample([e for e in triplets_to_sample_on if e['subject'] not in subject_filter_lookup[curr_subject]], 1)[0])
                new_edge['predicate'] = deepcopy(curr_triplet['predicate'])
                new_edge['object'] = deepcopy(curr_triplet['object'])
                new_edge['validity'] = 'never true'
            output.append(new_edge)

        return output

    def connect_negative_shuffle_subject(self, shuffle, shuffle_on, tar_tag, avoid_mixing=None):
        logger.info(f'connect_negative_shuffle_subject {tar_tag}')
        # We assume shuffle_on is only one field (usueally predicate or object)
        # Finding "clusters" that may not be shuffled internally when producing negative examples
        # (because the have downword monotone relations)
        connect_to = deepcopy(shuffle)
        triplets_to_shuffle_df = pd.DataFrame(([e[shuffle_on] for e in shuffle]))
        field_to_shuffle_counts = triplets_to_shuffle_df['subject'].value_counts()
        subjects_to_shuffle = set(triplets_to_shuffle_df['subject'])
        remaining_inds_to_choose = set(triplets_to_shuffle_df.index)

        for curr_subject, size in field_to_shuffle_counts.iteritems():
            potential_target_inds = deepcopy(remaining_inds_to_choose)
            tar_subjects = subjects_to_shuffle - {curr_subject}
            tar_subjects -= {e['subject'] for e in TeachAIKB().sample({'predicate': 'hypernym', 'object': curr_subject})}

            if avoid_mixing is not None and 'co-hyponyms' in avoid_mixing:
                subject_is_hyponym_of = {e['object'] for e in TeachAIKB().sample({'subject': curr_subject, 'predicate': 'hypernym'})}
                tar_subjects -= {e['subject'] for e in
                                      TeachAIKB().sample({'predicate': 'hypernym', 'object': list(subject_is_hyponym_of)})}

            if avoid_mixing is not None and 'co-meronyms' in avoid_mixing:
                subject_is_meronym_of = {e['subject'] for e in self.sample({'predicate': 'meronym', 'object': curr_subject})}
                tar_subjects -= {e['object'] for e in self.sample({'predicate': 'meronym', 'subject': list(subject_is_meronym_of)})}

            potential_target_inds &= set(triplets_to_shuffle_df[triplets_to_shuffle_df['subject'].isin(tar_subjects)].index)
            targets = [e for e in connect_to if e[shuffle_on]['subject'] == curr_subject]
            selected_inds = []
            for i in random.sample(potential_target_inds, len(potential_target_inds)):
                new_edge = {'subject': curr_subject,
                            'predicate': triplets_to_shuffle_df.loc[i, 'predicate'],
                            'object': triplets_to_shuffle_df.loc[i, 'object']}
                # checking if there is no triplet that is true with the same values:
                matching_edges_in_kb = self.lookup(new_edge)
                if len(matching_edges_in_kb) == 0:
                    targets[len(selected_inds)][tar_tag] = new_edge
                    targets[len(selected_inds)][tar_tag].update({'validity': 'never true'})
                    selected_inds.append(i)
                    if len(selected_inds) >= len(targets):
                        break
            if len(selected_inds) < len(targets):
                logger.debug(f'did not find enough for {curr_subject}: {len(selected_inds)} found, {len(targets)} required')
            else:
                logger.debug(f'{curr_subject}: {len(selected_inds)} found.')

            remaining_inds_to_choose -= set(selected_inds)

        return connect_to

    def sample_distractors(self, examples, sample, tar_tag):

        # building indexes:
        for i, sample_props in enumerate(sample):
            src_tag, src_fields, sample, exactly_sample_num, connect, balance_with_statement = sample_props

            # creating general indexes
            indexes = {}
            for field in ['subject', 'predicate', 'object', 'validity']:
                indexes[field] = {}
                for i, r in enumerate(examples):
                    if r[src_tag][field] not in indexes[field]:
                        indexes[field][r[src_tag][field]] = {i}
                    else:
                        indexes[field][r[src_tag][field]] |= {i}

            # Link the connection to existing tags.
            for i, example in tqdm(enumerate(examples), desc=f'adding distractors for {sample_props}'):
                cand_inds_signed = {}
                # the index helps us get candidates fast from the df of candidate_edges
                cand_inds = set(range(len(examples)))
                for field in src_fields:
                    cand_inds &= indexes[field][example[src_tag][field]]

                # making sure cand edges do not contain a duplicate of the currect example
                same_as_example_inds = indexes['subject'][example[src_tag]['subject']] & \
                                       indexes['predicate'][example[src_tag]['predicate']] & \
                                       indexes['object'][example[src_tag]['object']]
                cand_inds -= same_as_example_inds

                cand_inds_signed = {'always true':set(), 'never true': set()}
                for validity in ['always true', 'never true']:
                    if validity in indexes['validity']:
                        cand_inds_signed[validity] |= cand_inds & indexes['validity'][validity]

                if exactly_sample_num:
                    num_to_sample = sample
                else:
                    num_to_sample = random.sample(range(min(len(cand_inds_signed['always true']) + \
                                                            len(cand_inds_signed['never true']), sample) + 1), 1)[0]

                # Here we choose what is the validity value of the distractor we want to sample
                if balance_with_statement is not None:
                    # balance_with_statement is not None, that means we care about the validity value balancing.
                    validities_to_sample = {'always true': math.ceil(num_to_sample / 2), 'never true': math.ceil(num_to_sample / 2)}
                    if balance_with_statement and validities_to_sample[example[src_tag]['validity']] > 0:
                        validities_to_sample[example[src_tag]['validity']] -= 1
                else:
                    # Here we just randomly sample from a certain validity value (balance_with_statement is None, so it doesn't matter to us)
                    validities_to_sample = {'always true': 0, 'never true': 0}
                    validity_value_to_sample = random.sample(['always true', 'never true'],1)[0]
                    validities_to_sample[validity_value_to_sample] =  num_to_sample

                balanced_cand_inds = []
                for validity, num_to_sample in validities_to_sample.items():
                    if len(cand_inds_signed[validity]) >= num_to_sample:
                        balanced_cand_inds += random.sample(cand_inds_signed[validity], num_to_sample)

                # now actually sampling the rule we want to add to distractors
                if tar_tag not in example:
                    example[tar_tag] = {}
                for ind in balanced_cand_inds:
                    for tag in connect:
                        if tag not in example[tar_tag]:
                            example[tar_tag][tag] = []
                        example[tar_tag][tag].append(examples[ind][tag])

        return examples

    def print_stats(self):
        for part in ['statement', 'implicit_rule', 'property']:
            entities = {'dev': [], 'train': []}
            for e in self.examples_meta:
                if part in e:
                    if e['split'] == 'dev':
                        entities['dev'] += [e[part]['subject'], e[part]['object']]
                    elif e['split'] == 'train':
                        entities['train'] += [e[part]['subject'], e[part]['object']]
            if len(entities['dev']) == 0 | len(entities['train']) == 0:
                logger.info(f" {part} was not found or ablated.")
                continue

            entities_intersection_ratio = len(set(entities['dev']) & set(entities['train'])) / \
                                          len(set(entities['dev']) | set(entities['train']))
            logger.info(f"Dev/Train entity intersection in {part} :\n{entities_intersection_ratio}\n")
            if entities_intersection_ratio > 0.01:
                entity_stats = pd.DataFrame(
                    {'dev': pd.Series(entities['dev']).value_counts(), 'train': pd.Series(entities['train']).value_counts()}).dropna()
                entity_stats['min'] = entity_stats[['dev', 'train']].min(axis=1)
                logger.info(f"mutual entities stats:\n{entity_stats.sort_values(by='min')}")

        if 'statement' in self.examples_meta[0]:
            agg = pandas_multi_column_agg(pd.DataFrame([{'predicate': e['statement']['predicate'],'split':e['split'], 'z': 1} \
                                                    for e in self.examples_meta]), ['split', 'predicate'])
            logger.info(f"Predicate count per split:\n{agg}\n")

        examples_meta_df = pd.DataFrame(self.examples_meta)
        logger.info(f"Positive vs Negative:\n{pandas_multi_column_agg(examples_meta_df, ['split', 'answer'])}\n")



