# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
import os
from LeapOfThought.artisets.soft_reasoning.hypernyms import Hypernyms

class TestHypernyms:

    @pytest.mark.parametrize("variant", ("statement_only", "explicit_only", "implicit_only", "statement_only_no_context",
                                         "training_mix", "statement_subject_lang_selectivity"))
    def test_build_artificial_dataset_all(self, variant):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-c", "--artiset_name", type=str, help="The name of the artiset class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-v", "--variant", type=str, help="The task stage to run")
        parse.add_argument("-e", "--experiment_version", type=str, help="mini version, appended as a suffix to the artiset file name", default="")
        parse.add_argument("-s", "--split", type=str, help="Build a specific split", default=None)
        parse.add_argument("-fs", "--split_by_field", type=str,
                           help="dev+test vs train split so that the specified field would be disjoint", default="")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("--config_path", type=str, help="Artisets config file", default="config.json")
        parse.add_argument("--save_sample", action='store_true', help="Saves a sample jsonl", default=False)
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-c", "Hypernyms","-o", "create_artiset","-v", variant, "-s", "dev", \
                                 "-out","data/artiset_samples/Hypernyms/sample.jsonl","--save_sample"])

        if not os.path.exists("data/artiset_samples/Hypernyms"):
            os.mkdir('data/artiset_samples/Hypernyms')

        artiset = Hypernyms(args)

        # reducing data size to a sample:
        artiset._config['test_dev_size'] = [0, 0]
        artiset._config['max_number_of_examples'] = 50

        artiset.build_artificial_dataset(args)


    @pytest.mark.parametrize("variant", (["training_mix"]))
    def test_build_artificial_dataset_single(self, variant):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-c", "--artiset_name", type=str, help="The name of the artiset class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-v", "--variant", type=str, help="The task stage to run")
        parse.add_argument("-e", "--experiment_version", type=str, help="mini version, appended as a suffix to the artiset file name", default="")
        parse.add_argument("-s", "--split", type=str, help="Build a specific split", default=None)
        parse.add_argument("-fs", "--split_by_field", type=str,
                           help="dev+test vs train split so that the specified field would be disjoint", default="")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("--config_path", type=str, help="Artisets config file", default="config.json")
        parse.add_argument("--save_sample", action='store_true', help="Saves a sample jsonl", default=False)
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-c", "Hypernyms", "-o", "create_artiset", "-v", variant, \
                                 "-out", "data/artiset_samples/Hypernyms/sample.jsonl", "--save_sample"])

        # , "-s", "dev"
        if not os.path.exists("data/artiset_samples/Hypernyms"):
            os.mkdir('data/artiset_samples/Hypernyms')

        artiset = Hypernyms(args)

        # reducing data size to a sample:
        artiset._config['test_dev_size'] = [900, 0]
        artiset._config['max_number_of_examples'] = 50

        artiset.build_artificial_dataset(args)
