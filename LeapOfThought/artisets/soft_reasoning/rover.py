
import random
import json
import logging

from tqdm import tqdm
import pandas as pd
import hashlib
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.common.file_utils import cached_path


logger = logging.getLogger(__name__) # pylint: disable=invalid-name

class Rover(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'Rover'
        logger.info("loading...")
        super().__init__(args)

        self._rover = {'train':[],'dev':[]}
        for split in ['train','dev']:
            soft_reasoning_dev_path = cached_path('https://aigame.s3-us-west-2.amazonaws.com/data/resources/soft_reasoning_' \
                                                  + split + '.jsonl')
            with open(soft_reasoning_dev_path) as f:
                for line in f:
                    try:
                        self._rover[split].append(json.loads(line))
                    except:
                        print('problem extracting line ')

    def build_artificial_dataset(self, args):

        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []

        logger.info("building examples")

        for split in ['train', 'dev']:
            for example in tqdm(self._rover[split]):

                example['split'] = split
                # append_teachyourai_format_example() is method implemented in ArtiSet class and takes an example dict
                # (that must contain a "phrase", "answer") and converts it to a BooleanQA format
                self.append_teachyourai_format_example(example, do_print=self._config['debug'])

                self.examples_meta.append(example)

                if self._config['max_number_of_examples'] != -1 and \
                        len(self.artiset_data) >= self._config['max_number_of_examples']:
                    break

        self.examples_meta = pd.DataFrame(self.examples_meta)

        # save_dataset() is a is method implemented in ArtiSet class that automatically saves the artiset
        # if the config output_file contains the string _sample.jsonl it will be saved in a more readable format
        # otherwise it will split the examples in self.artiset_data into train, dev, test and save them in s3
        # if output_file startswith s3:// otherwise locally. (If output_file is empty, it will not save)
        self.save_dataset()






