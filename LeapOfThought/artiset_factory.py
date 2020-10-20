import os, json
import sys
import gzip, random
from setuptools import find_packages
from pkgutil import iter_modules
from LeapOfThought.common.file_utils import upload_jsonl_to_s3, cached_path

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ArtisetFactory:
    def __init__(self):
        pass

    def upper_to_lower_notation_name(self, artiset_name):
        return ''.join(['_' + c.lower()  if c.isupper() else c for c in artiset_name ])[1:]

    def find_artiset(self, path, callange_to_find):
        modules = list()
        for pkg in [''] + find_packages(path):
            pkgpath = path + '/' + pkg.replace('.', '/')
            if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
                for _, name, ispkg in iter_modules([pkgpath]):
                    if not ispkg:
                        modules.append(pkg + '.' + name)
            else:
                for info in iter_modules([pkgpath]):
                    if not info.ispkg:
                        modules.append(pkg + '.' + info.name)

        found_artiset = [module for module in modules if module.find('.' + callange_to_find) > -1]
        if len(found_artiset) > 0:
            found_artiset = found_artiset[0]
            if found_artiset.startswith('.'):
                found_artiset =  found_artiset[1:]
        else:
            found_artiset = None

        return found_artiset

    def get_artiset(self, artiset_name, args):
        artiset_name_lower = self.upper_to_lower_notation_name(artiset_name)
        module_name = self.find_artiset(os.path.dirname(os.path.abspath(__file__)) + '/artisets', artiset_name_lower)
        try:
            mod = __import__('LeapOfThought.artisets.' + module_name, fromlist=[artiset_name])
        except:
            raise ValueError('artiset_name not found!')

        return getattr(mod, artiset_name)(args)

    def create_new_artiset(self, artiset_name, artiset_module, copy_from, args):
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/artisets')
        copy_from_lower = self.upper_to_lower_notation_name(copy_from)
        artiset_name_lower = self.upper_to_lower_notation_name(artiset_name)
        copy_from_module = self.find_artiset(os.getcwd(), copy_from_lower)
        if copy_from_module is None:
            assert (ValueError('copy_from artiset not found!'))
        copy_from_path = copy_from_module.replace('.',os.sep) + '.py'

        if not os.path.isdir(artiset_module):
            os.mkdir(artiset_module)
            open(os.path.join(artiset_module,'__init__.py'), 'a').close()

        with open(copy_from_path,'r') as f:
            copied_artiset_txt =  f.read()
            copied_artiset_txt = copied_artiset_txt.replace(copy_from, artiset_name)

        if len(artiset_module) > 0:
            new_artiset_path = os.path.join(artiset_module, artiset_name_lower) + '.py'
        else:
            new_artiset_path = artiset_name_lower + '.py'
        with open(new_artiset_path, 'w') as f:
            f.write(copied_artiset_txt)

        # duplicating the test
        os.chdir('../../tests/artisets')
        if not os.path.isdir(artiset_module):
            os.mkdir(artiset_module)
        with open(copy_from_path.replace('.py','_test.py'),'r') as f:
            copied_artiset_txt =  f.read()
            copied_artiset_txt = copied_artiset_txt.replace('artisets.' + copy_from_module, \
                                                                'artisets.' + artiset_module + '.' + artiset_name_lower)
            copied_artiset_txt = copied_artiset_txt.replace(copy_from, artiset_name)
            copied_artiset_txt = copied_artiset_txt.replace(copy_from_lower, artiset_name_lower)

        if len(artiset_module) > 0:
            new_artiset_path = os.path.join(artiset_module, artiset_name_lower) + '_test.py'
        else:
            new_artiset_path = artiset_name_lower + '_test.py'
        with open(new_artiset_path, 'w') as f:
            f.write(copied_artiset_txt)

        # adding to config file:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path) , 'r') as f:
            config = json.load(f)
        config[artiset_name] = config[copy_from]
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path) , 'w') as f:
            json.dump(config, f ,indent=4)

    def combine_artisets(self, s3_subdirs, s3_prefix,  sample_sizes, dev_sample_sizes, output_file, args):

        random.seed(17)
        for split, sample_sizes in zip(['dev', 'train'], [dev_sample_sizes, sample_sizes]):
            merged_examples = []
            for filename_to_merge, sample_size in zip(s3_subdirs.split(','), sample_sizes.split(',')):

                logger.info(f'\n-------Merging file {filename_to_merge}-----------\n')
                chunk_dataset_path = cached_path(
                    s3_prefix + '/' + filename_to_merge + '_' + split + '.jsonl.gz')

                examples = []
                with gzip.open(chunk_dataset_path, 'r') as f:
                    for line in f:
                        examples.append(json.loads(line))

                print(f"{len(examples)} found for {filename_to_merge} {split}, sampling {sample_size}")
                if int(sample_size) >= len(examples):
                    merged_examples += examples
                else:
                    merged_examples += random.sample(examples, int(sample_size))

            random.shuffle(merged_examples)

            print(f"Total of {len(merged_examples)} examples saved for {split}")
            upload_jsonl_to_s3(output_file + '_' + split + '.jsonl.gz', merged_examples)





