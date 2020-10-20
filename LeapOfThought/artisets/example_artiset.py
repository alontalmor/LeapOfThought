
import random
import json
import logging

from tqdm import tqdm
import pandas as pd
from LeapOfThought.artiset import ArtiSet
from LeapOfThought.common.file_utils import cached_path


logger = logging.getLogger(__name__) # pylint: disable=invalid-name

class ExampleArtiSet(ArtiSet):
    def __init__(self, args):
        self.artiset_name = 'ExampleArtiset'
        logger.info("loading...")
        super().__init__(args)

    def build_artificial_dataset(self, args):

        # examples_meta is a pandas DataFrame that contain all examples with additional meta data for
        # the task, and will be automatically save as "..._meta.jsonl" file with the artiset files
        self.examples_meta = []

        logger.info("building examples")

        for example_ind in tqdm(range(self._config['max_number_of_examples'])):

            # We may have more than one variant to our artiset
            if args.variant == '100':
                largest_num = random.randint(1, 100)
                max_distractor_diff = 10
            elif args.variant == '100000':
                largest_num = random.randint(1, 100000)
                max_distractor_diff = 100

            # For the example we will just create a number comparison artiset
            answer = str(largest_num)
            dist1 = str(largest_num - random.randint(1, max_distractor_diff))

            question = 'Is ' + answer +  ' larger than ' + dist1 + '?'
            example = {'phrase':question , \
                       'answer': 1}

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






