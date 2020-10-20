
import random
import json
import logging

from tqdm import tqdm
import pandas as pd
import hashlib
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.common.file_utils import cached_path
from LeapOfThought.resources.conceptnet import ConceptNet as ConceptNetAPI

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

class ConceptNet(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'ConceptNet'
        logger.info("loading...")
        super().__init__(args)

    def edgefilter_create_obqa_dataset(self, data, name='', do_print=False, save_file=False, sanity=False, paraphrase=True):

        relation_dict = {'/r/RelatedTo': 'is related to',
                         '/r/Antonym': 'is the opposite of',
                         '/r/AtLocation': 'is usually located at',
                         '/r/Causes': 'usually causes',
                         '/r/HasPrerequisite': 'has a prerequisite of',
                         '/r/PartOf': 'is part of',
                         '/r/HasFirstSubevent': 'has a first subevent of',
                         '/r/HasLastSubevent': 'has a last subevent of',
                         '/r/MotivatedByGoal': 'is motivated by a goal of',
                         '/r/UsedFor': 'is usually used for',
                         '/r/HasSubevent': 'has a subevent of',
                         '/r/NotDesires': 'does not desire',
                         '/r/Desires': 'desires',
                         '/r/CapableOf': 'is capable of',
                         '/r/CapableOf': 'is capable of',
                         '/r/CausesDesire': 'causes desire of',
                         '/r/DistinctFrom': 'is distinct from',
                         '/r/ReceivesAction': 'usually recieves an action of',
                         '/r/InstanceOf': 'is an instance of',
                         '/r/CreatedBy': 'is created by',
                         '/r/DefinedAs': 'is defined as',
                         '/r/LocatedNear': 'is located near',
                         '/r/NotCapableOf': 'is not capable of',
                         '/r/NotHasProperty': 'does not have a property of',
                         '/r/HasA': 'has',
                         '/r/Entails': 'entails',
                         '/r/HasProperty': 'has a property of'}

        openbookqa_format = []
        for qid, example in data.iterrows():
            if example['relation'] in relation_dict:
                question = example['start_term'] + ' ' + relation_dict[example['relation']] + ' ' + example['end_term']
                openbookqa_format.append({'answer': example['class'], 'id': qid, \
                                          'phrase': question})

        random.seed(0)
        random.shuffle(openbookqa_format)
        return openbookqa_format

    def build_artificial_dataset(self, args):

        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []

        logger.info("building examples")

        cn_edges = ConceptNetAPI().get_ConceptNet_full()
        cn_edges_index = cn_edges.set_index(['start_term', 'relation', 'end_term'])

        cn_edges_cent = cn_edges[(cn_edges['end_term'].fillna('').apply(len) > 2) & \
                                 (cn_edges['start_term'].fillna('').str.len() > 2)]
        cn_edges_cent = cn_edges_cent[cn_edges_cent['weights'] > 1]
        cn_edges_cent = cn_edges_cent[(cn_edges_cent['relation'] != '/r/IsA') & \
                                      (cn_edges_cent['relation'] != '/r/DerivedFrom') & \
                                      (cn_edges_cent['relation'] != '/r/RelatedTo') & \
                                      (cn_edges_cent['relation'] != '/r/FormOf') & \
                                      (cn_edges_cent['relation'] != '/r/Synonym') & \
                                      (cn_edges_cent['relation'] != '/r/SimilarTo') & \
                                      (cn_edges_cent['relation'] != '/r/MannerOf') & \
                                      (cn_edges_cent['relation'] != '/r/HasContext') & \
                                      (cn_edges_cent['relation'] != '/r/EtymologicallyRelatedTo') & \
                                      ~(cn_edges_cent['relation'].str.contains('/r/dbpedia'))]

        for id, example in tqdm(cn_edges_cent.iterrows(), total=len(cn_edges_cent)):
            m = hashlib.md5()
            m.update(example['start_term'].encode())
            m.update(example['end_term'].encode())
            m.update(example['relation'].encode())
            qid = m.hexdigest()
            cn_edges_cent.at[id, 'qid'] =  qid
        cn_edges_cent = cn_edges_cent.set_index('qid')

        cn_edges_cent_neg = cn_edges_cent.copy(deep=True)
        for relation in list(set(cn_edges_cent['relation'])):
            cn_edges_cent_neg.loc[cn_edges_cent_neg['relation'] == relation, 'end_term'] = \
                list(cn_edges_cent_neg.loc[cn_edges_cent_neg['relation'] == relation, 'end_term'].sample(frac=1))

        # filterering the terms
        print(len(cn_edges_cent_neg))
        lookup = set(cn_edges_index.index)
        ids_to_take = []
        for id, edge in cn_edges_cent_neg.iterrows():
            if (edge['start_term'], edge['relation'], edge['end_term']) not in lookup:
                ids_to_take.append(id)
        cn_edges_cent_neg = cn_edges_cent_neg.loc[ids_to_take]
        print(len(cn_edges_cent_neg))

        # changing the IDs for the negative examples
        cn_edges_cent_neg.index = cn_edges_cent_neg.index + '_1'

        # assigning the classes for the positive and negative examples
        cn_edges_cent['class'] = 1
        cn_edges_cent_neg['class'] = 0

        cut_point = 37000
        all_cn_data = cn_edges_cent[0:cut_point].append(cn_edges_cent_neg[0:cut_point])
        for example in self.edgefilter_create_obqa_dataset(all_cn_data):
            example['split'] = 'train'
            self.append_teachyourai_format_example(example, do_print=self._config['debug'])
            self.examples_meta.append(example)

        all_cn_data = cn_edges_cent[cut_point:].append(cn_edges_cent_neg[cut_point:])
        for example in self.edgefilter_create_obqa_dataset(all_cn_data):
            example['split'] = 'dev'
            self.append_teachyourai_format_example(example, do_print=self._config['debug'])
            self.examples_meta.append(example)


        self.examples_meta = pd.DataFrame(self.examples_meta)

        # save_dataset() is a is method implemented in ArtiSet class that automatically saves the artiset
        # if the config output_file contains the string _sample.jsonl it will be saved in a more readable format
        # otherwise it will split the examples in self.artiset_data into train, dev, test and save them in s3
        # if output_file startswith s3:// otherwise locally. (If output_file is empty, it will not save)
        self.save_dataset()






