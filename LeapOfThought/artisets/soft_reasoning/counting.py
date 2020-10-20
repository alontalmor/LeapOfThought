import random
from copy import deepcopy
import logging
import json, gzip
from tqdm import tqdm
import pandas as pd
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.resources.teachai_kb import TeachAIKB
from LeapOfThought.common.data_utils import pandas_multi_column_agg
from LeapOfThought.common.file_utils import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Counting(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'Counting'
        self.variant = args.variant
        self.experiment_version = args.experiment_version
        logger.info("loading...")
        super().__init__(args)

    def print_stats(self):
        super().print_stats()

        # counting specific stats
        meta = pd.DataFrame(self.examples_meta)
        logger.info(f"Is_hypothetical_statement :\n{pandas_multi_column_agg(meta, ['split', 'is_hypothetical_statement'])}\n")

        logger.info(f"Duplicate statements :\n{len(meta['phrase']) - len(set(meta['phrase']))}\n")
        # analyzing the distibution of total_num_of_instances and % counted_instances
        if 'counted_instances' in meta:
            meta['counted_instances'] = meta['counted_instances'].fillna('')
            # meta = meta[meta['counted_instances'].fillna('')]
            meta['counted_instances_perc'] = meta['counted_instances'].apply(len) \
                                             / meta['total_num_of_instances']
            meta = meta[meta['is_hypothetical_statement'] == False]
            logger.info(f"counted_instances_perc mean answer :\n{meta.groupby('counted_instances_perc')['answer'].mean().round(2)}\n")
            logger.info(f"counted_instances_perc count :\n{meta.groupby('counted_instances_perc')['answer'].count()}\n")

            # calculating total number of instances per predicate
            agg = pandas_multi_column_agg(pd.DataFrame([{'predicate': e['statement']['predicate'], \
                                                         'total_num_of_instances': e['total_num_of_instances'], 'z': 1} \
                                                        for e in self.examples_meta if 'total_num_of_instances' in e]), \
                                          ['predicate', 'total_num_of_instances'])
            logger.info(f"total_num_of_instances pre predicate:\n{agg}\n")

            # calculating total number of instances per predicate
            meta['predicate'] = meta['statement'].apply(lambda x: x['predicate'])
            agg = pandas_multi_column_agg(meta[['total_num_of_instances', 'counted_instances_perc', 'answer', 'phrase']], \
                                          ['total_num_of_instances', 'counted_instances_perc'])
            logger.info(f"counted_instances_perc pre predicate:\n{agg}\n")

    def building_non_balanced_counting_examples(self, counted_instances, false_statements, split):
        examples = []
        for counted_instances, false_statements in zip(counted_instances.to_list(), false_statements.to_list()):
            counted_instances, total_num_of_instances = counted_instances

            # for each set of counted_instances and false_statements we now choose how many of the
            # counted instances we should remove before choosing the statment (where 0 means we are keeping all
            # the counted instances)
            all_counted_instaces = deepcopy(counted_instances)
            counted_instances_in_context = []
            total_count_rule = {'subject': counted_instances[0]['object'], 'predicate': 'has ' + str(total_num_of_instances),
                            'object': counted_instances[0]['predicate'], 'validity': 'always true'}
            for num_of_counted_instances in range(total_num_of_instances + 1):

                if total_num_of_instances == 1 and len(counted_instances_in_context) == 0 and random.sample([0, 0, 1], 1)[0]:
                    instances_to_sample_statement = counted_instances
                else:
                    instances_to_sample_statement = counted_instances + [false_statements.pop()]
                # in the dev set, for balancing labels, let's take cases in which the count reaches the max.
                if split == 'dev' and num_of_counted_instances == total_num_of_instances:
                    num_to_sample = random.sample([0, 1, 1], 1)[0]
                else:
                    # usually sample 2 statement for  each num_of_counted_instances
                    num_to_sample = random.sample([1] + [min(len(instances_to_sample_statement), 2)] * 2, 1)[0]

                statements = random.sample(instances_to_sample_statement, num_to_sample)

                for statement in statements:
                    if self.variant == 'increment_counted_instances':
                        for increment_counted_instances_num in range(total_num_of_instances + 1):
                            if statement['validity'] == 'always true' and increment_counted_instances_num == total_num_of_instances:
                                continue
                            examples.append({'statement': deepcopy(statement),
                                             'counted_instances': deepcopy(all_counted_instaces[0:increment_counted_instances_num]),
                                             'total_num_of_instances': total_num_of_instances,
                                             'total_count_rule': total_count_rule})
                    else:
                        examples.append({'statement': deepcopy(statement),
                                         'counted_instances': deepcopy(counted_instances_in_context),
                                         'total_num_of_instances': total_num_of_instances,
                                         'total_count_rule': total_count_rule})

                if len(counted_instances) > 0:
                    counted_instances_in_context.append(counted_instances.pop())

        return examples

    def read_predictions_and_meta(self, name, variant, base_model, model, full_path=None):

        if full_path is None:
            pred_path = cached_path("https://aigame.s3-us-west-2.amazonaws.com/predictions/" +
                                    base_model + "/" + model + "/" + name + "_" + variant + ".json")
        else:
            pred_path = full_path

        preds = []
        with open(pred_path) as f:
            all_results = json.load(f)
            print(f"total EM for {base_model} {model} is {all_results['EM']}")
            preds = all_results['predictions']

        preds = pd.DataFrame(preds)
        pred_path = cached_path("https://aigame.s3-us-west-2.amazonaws.com/data/" + \
                                name + "/" + name + "_" + variant + "_meta.jsonl.gz")

        metadata = []
        with gzip.open(pred_path) as f:
            for line in f:
                metadata.append(json.loads(line))

        # df =  pd.DataFrame(preds).merge(metadata,on='id',how='inner')
        metadata_df = pd.DataFrame(metadata)
        df = pd.merge(preds, metadata_df, on=['id'], how="outer", indicator=True)
        df = df[df['_merge'] == 'both']
        df = df.loc[:, ~df.columns.str.contains('_y')]
        df.columns = df.columns.str.replace('_x', '')
        return df

    def building_balanced_counting_examples(self, counted_instances, false_statements, split):

        # loading predictions for single counted instances:
        if self.variant in ['increment_counted_instances_prob_asc', 'increment_counted_instances_prob_desc']:
            curr_model_name = 'counting_training_mix_no_hypothetical_38700_3'
            curr_model = 'roberta-large'
            statement_only_predictions = self.read_predictions_and_meta('counting', 'statement_only', curr_model, curr_model_name)

        examples = []
        for counted_instances, false_statements in tqdm(zip(counted_instances.to_list(), false_statements.to_list())):
            counted_instances, total_num_of_instances = counted_instances

            # for each set of counted_instances and false_statements we now choose how many of the
            # counted instances we should remove before choosing the statment (where 0 means we are keeping all
            # the counted instances)
            total_count_rule = {'subject': counted_instances[0]['object'], 'predicate': 'has ' + str(total_num_of_instances),
                            'object': counted_instances[0]['predicate'], 'validity': 'always true'}
            all_counted_instaces = deepcopy(counted_instances)
            all_false_statements = deepcopy(false_statements)
            for num_counted_instances_to_remove in range(len(counted_instances) + 1):
                # Creating one positive and one negative example for each count (assuming the counted_instances and
                # false_statements are randomized):
                for type in ['positive', 'negative']:
                    statement = None
                    if type == 'positive' and num_counted_instances_to_remove > 0:
                        statement = counted_instances.pop()
                    elif type == 'negative':  # and (split == 'train' or random.sample([1,1,1,0,0],1)[0]):
                        statement = false_statements.pop()

                    if statement is not None:
                        # In the increment_counted_instances versions, we just systematically add the counted instances in a certain order
                        # from zero to the max number of instances (depending on the validity of the statement)
                        if self.variant in ['increment_counted_instances', 'increment_counted_instances_prob_asc', 'increment_counted_instances_prob_desc']:
                            for increment_counted_instances_num in range(total_num_of_instances + 1):
                                if self.variant == 'increment_counted_instances':
                                    increment_counted_instances = deepcopy(all_counted_instaces[0:increment_counted_instances_num])
                                elif self.variant in ['increment_counted_instances_prob_asc', 'increment_counted_instances_prob_desc']:
                                    probabilities = pd.Series([statement_only_predictions[statement_only_predictions['statement'] == ci].iloc[0]['label_probs'][1]
                                         for ci in all_counted_instaces])
                                    if self.variant == 'increment_counted_instances_prob_asc':
                                        increment_counted_instances = deepcopy([all_counted_instaces[i] for i in list(
                                            probabilities.sort_values(ascending=True)[0:increment_counted_instances_num].index)])
                                    else:
                                        increment_counted_instances = deepcopy([all_counted_instaces[i] for i in list(
                                            probabilities.sort_values(ascending=False)[0:increment_counted_instances_num].index)])

                                if statement['validity'] == 'always true' and increment_counted_instances_num == total_num_of_instances:
                                    continue
                                examples.append({'statement': deepcopy(statement),
                                                 'counted_instances': increment_counted_instances,
                                                 'total_num_of_instances': total_num_of_instances,
                                                 'total_count_rule': total_count_rule})
                        else:
                            if self.experiment_version in ['count_reached_mix']:
                                new_counted_instances = deepcopy(random.sample([s for s in all_false_statements if s != statement] \
                                                       + counted_instances, len(counted_instances)))
                                for inst in new_counted_instances:
                                    inst['validity'] = 'always true'
                                examples.append({'statement': deepcopy(statement),
                                                 'counted_instances': new_counted_instances,
                                                 'total_num_of_instances': total_num_of_instances,
                                                 'total_count_rule': total_count_rule})

                            examples.append({'statement': deepcopy(statement),
                                             'counted_instances': deepcopy(counted_instances),
                                             'total_num_of_instances': total_num_of_instances,
                                             'total_count_rule': total_count_rule})

                    # adding a negative example for stroger signal when count is reached in some training variants.
                    if self.variant in ['training_mix_lv', 'training_mix_lv_oversample','training_mix_oversample', 'training_mix_lv_oversample_neg'] \
                            and num_counted_instances_to_remove == 0 and total_num_of_instances > 1:
                        statement = random.sample(all_false_statements,1)[0]
                        examples.append({'statement': deepcopy(statement),
                                         'counted_instances': deepcopy(counted_instances),
                                         'total_num_of_instances': total_num_of_instances,
                                         'total_count_rule': total_count_rule})


        return examples


    def build_artificial_dataset(self, args):
        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []
        random.seed(17)

        logger.info("building examples")

        # Querying all functional relations from KB.
        func_rel_triplets = TeachAIKB().sample({'predicate': ['super bowl loser', 'super bowl winner', 'band member', 'capital', \
                                                              'director', 'release year', 'founder', 'headquarter', 'child', 'spouse',
                                                              'CEO'], \
                                                'source': ['wikidata']},
                                               select=['subject', 'predicate', 'object', 'validity'])

        # split dev and train
        function_relations_split = {'dev': [], 'train': []}
        for e in func_rel_triplets:
            if sum(bytearray(e['object'].encode())) % 7 == 0 or e['object'] in ['Germany', 'United States of America', 'Israel']:
                function_relations_split['dev'].append(e)
            else:
                function_relations_split['train'].append(e)

        for split, function_relations in function_relations_split.items():
            if args.split is not None and split != args.split:
                continue

            logger.info(f'---------------  {split} ------------------')

            function_relations_neg = self.self_negative_subject_sample(function_relations, avoid_mixing='predicates', \
                                                                       over_sample=2)

            # In counting we have "counted instances" which are the actual instances to be counted
            # these are collected using aggregation of predicate and object.
            counted_instances = pd.DataFrame(function_relations).groupby(['predicate', 'object']).apply(lambda x: \
                                                                                                            (x.to_dict(orient='rows'),
                                                                                                             len(x)))
            false_statements = pd.DataFrame(function_relations_neg).groupby(['predicate', 'object']).apply(lambda x: \
                                                                                                               x.to_dict(orient='rows'))

            examples = self.building_balanced_counting_examples(counted_instances, false_statements, split)

            # creating the statement
            # creating implicit_rule (implicit rules in counting are "entitiy is a company")
            # examples = TeachAIKB().connect(connect_to=statements_with_instances,
            #                                                 constraints={'validity': 'always true', 'predicate': ['hypernym']}, \
            #                                                 src_tags=['statement'], connection_point=[{'object': 'subject'}],
            #                                                 tar_tag='implicit_rule')

            # creating the hypernym count property (every company has 2 founders)
            # for e in examples:
            #    e['hypernym_count_property'] = {'subject': e['implicit_rule']['object'], 'predicate': 'has 1',
            #                         'object': e['statement']['predicate'], 'validity': 'always true'}

            if args.variant in ['training_mix_lv_oversample_neg','training_mix_lv_oversample','training_mix_oversample']:
                examples = pd.Series(examples).sample(frac=1.5, random_state=17, replace=True).to_list()


            # Sampling distractors.  "[(src_tag, src_fields, num_to_sample, exactly_sample_num, fields_to_take, balance_with_statement)]
            if args.variant in ['training_mix_lv', 'training_mix_lv_oversample', 'training_mix_lv_oversample_neg','training_mix_oversample']:
                self.sample_distractors(examples, tar_tag='distractors', sample=[
                    ('statement', ['predicate'], 1, True, ['counted_instances', 'total_count_rule'], None),
                    ('statement', [], 2, False, ['counted_instances', 'total_count_rule'], None)])
            else:
                self.sample_distractors(examples, tar_tag='distractors', sample=[
                    ('statement', ['predicate'], 2, True, ['counted_instances', 'total_count_rule'], False),
                    ('statement', [], 2, False, ['counted_instances', 'total_count_rule'], False)])
            # ('implicit_rule', ['predicate'], 2, False, ['implicit_rule'], False)])

            # Hypothetical statements vs real fact statements
            hypothetical_portion = {'training_mix': 0, 'training_mix_with_hypothetical': 0.3, 'hypothetical_only': 1, \
                                    'hypothetical_true_label': 1}
            if args.variant not in hypothetical_portion:
                hypothetical_portion[args.variant] = 0
            # hypothtical_inds = random.sample(range(len(examples)), int(hypothetical_portion[args.variant] * len(examples)))
            hypothtical_inds = pd.Series(range(len(examples))).sample(frac=hypothetical_portion[args.variant], random_state=17).to_list()
            for i, e in enumerate(examples):
                if i in hypothtical_inds:
                    e['is_hypothetical_statement'] = True
                    # In the hypothetical case, when the instance count is strictly less than the total number of instances,
                    # the hypothetical_statement becomes true.
                    if len(e['counted_instances']) < e['total_num_of_instances'] and not args.variant == 'hypothetical_true_label':
                        e['statement']['validity'] = 'always true'
                else:
                    e['is_hypothetical_statement'] = False

            # mixing 10% of the statements with no context in the training mix only.
            if args.variant in ['training_mix', 'training_mix_lv', 'training_mix_lv_oversample','training_mix_lv_oversample_neg',\
                                'training_mix_with_hypothetical','training_mix_oversample']:
                examples += [{'statement': e['statement']} for e in random.sample([e for e in examples \
                                if e['statement']['validity'] == 'never true'], int(0.05 * float(len(examples))))]
                examples += [{'statement': e['statement']} for e in random.sample([e for e in examples \
                                if e['statement']['validity'] == 'always true'], int(0.05 * float(len(examples))))]

            # for each variation, the proportion in which each rule type will be filtered.
            ablations = {
                'training_mix': [(['distractors'], 0.2), (['total_count_rule'], 0.05)],
                'training_mix_no_total': [(['distractors'], 0.2), (['total_count_rule'], 1)],
                'no_total': [(['total_count_rule'], 1)],
                'training_mix_lv': [(['distractors'], 0.2), (['total_count_rule'], 0.05)],
                'training_mix_oversample': [(['distractors'], 0.2), (['total_count_rule'], 0.05)],
                'training_mix_lv_oversample': [(['distractors'], 0.2), (['total_count_rule'], 0.05)],
                'training_mix_lv_oversample_neg': [(['distractors'], 0.2), (['total_count_rule'], 0.05)],
                'training_mix_with_hypothetical': [(['distractors'], 0.2), (['total_count_rule'], 0.05)],
                'statement_only': [(['counted_instances', 'total_count_rule'], 1), (['distractors'], 1)],
                'statement_only_with_distractors': [(['counted_instances', 'total_count_rule'], 1)],
            }
            if args.variant not in ablations:
                ablations[args.variant] = []

            if args.variant in ['no_total','training_mix_no_total']:
                for e in examples:
                    del e['distractors']['total_count_rule']

            # Actively splitting between test and dev (50/50)
            if split == 'dev':
                # making sure that for the same amount of examples the split will always be the same.
                random.seed(71)
                all_inds = [i for i in range(len(examples))]
                dev_inds = random.sample(all_inds, int(len(all_inds) / 2))
                test_inds = list(set(all_inds) - set(dev_inds))
                splits = [('dev', [examples[i] for i in dev_inds]),
                          ('test', [examples[i] for i in test_inds])]
            else:
                splits = [('train', examples)]

            for final_split, final_examples in splits:
                if args.experiment_version in ['count_reached','count_reached_mix']:
                    final_examples = [e for e in final_examples if len(e['counted_instances']) == e['total_num_of_instances']]
                elif args.experiment_version in ['count_not_reached']:
                    final_examples = [e for e in final_examples if len(e['counted_instances']) < e['total_num_of_instances']]
                elif args.experiment_version == 'between_one_and_full_counted_instances':
                    final_examples = [e for e in final_examples if len(e['counted_instances']) > 0 and \
                                len(e['counted_instances']) < e['total_num_of_instances']]


                self.build_statement_rule_property_examples(final_examples, split=final_split, \
                    rule_tags=['counted_instances', 'total_count_rule'],
                    ablate_same_distractor_fields=False, \
                    nlg_sampling=True if args.variant in ['training_mix_lv','training_mix_lv_oversample',\
                                                          'training_mix_lv_oversample_neg'] else False,\
                    ablation_list=ablations[args.variant], use_shorthand=True, \
                    reverse_validity_frac=0.3 if args.variant == 'training_mix_lv_oversample_neg' else 0)


        self.print_examples(30)
        self.print_stats()
        self.examples_meta = pd.DataFrame(self.examples_meta)
        self.save_dataset()






