
import random
import json
import logging
import gzip

from tqdm import tqdm
import pandas as pd
import hashlib
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.common.file_utils import cached_path


logger = logging.getLogger(__name__) # pylint: disable=invalid-name

class Sizes(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'Sizes'
        logger.info("loading...")
        super().__init__(args)


    def olmpics_bool_masked_dataset_to_yesno(self, dataset_name):
        output = []
        for split in ['train', 'dev']:
            coffee_cats_cached = cached_path('https://olmpics.s3.us-east-2.amazonaws.com/challenge/' + \
                                             dataset_name + '_' + split + '.jsonl.gz')
            bool_format_examples = []
            with gzip.open(coffee_cats_cached) as f:
                header = json.loads(f.readline())
                for line in f:
                    example = json.loads(line)
                    if example['answerKey'] == 'A':
                        answer1 = 1
                        answer2 = 0
                    else:
                        answer1 = 0
                        answer2 = 1

                    bool_format_examples.append({ \
                        'phrase': example['question']['stem'].replace('[MASK]', \
                        example['question']['choices'][0]['text']), 'answer': answer1,
                        'id': example['id'], 'split':split})
                    bool_format_examples.append({ \
                        'phrase': example['question']['stem'].replace('[MASK]',
                        example['question']['choices'][1]['text']), 'answer': answer2,
                        'id': example['id'] + '_1','split': split})
            output += bool_format_examples

        return output
    def build_artificial_dataset(self, args):

        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []

        logger.info("building examples")

        for example in tqdm(self.olmpics_bool_masked_dataset_to_yesno('size_comparison/size_comparison')):

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






