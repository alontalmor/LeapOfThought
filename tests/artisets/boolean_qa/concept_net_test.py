# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
import os
from LeapOfThought.artisets.boolean_qa.concept_net import ConceptNet

class TestExampleArtiSet:
    @pytest.mark.parametrize("variant", ([""]))
    def test_build_artificial_dataset(self, variant):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-c", "--artiset_name", type=str, help="The name of the artiset class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-v", "--variant", type=str, help="The task stage to run")
        parse.add_argument("-s", "--split", type=str, help="Build a specific split", default=None)
        parse.add_argument("-fs", "--split_by_field", type=str,
                           help="dev+test vs train split so that the specified field would be disjoint", default="")
        parse.add_argument("-e", "--experiment_version", type=str, help="mini version, appended as a suffix to the artiset file name", default="")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("--config_path", type=str, help="Artisets config file", default="config.json")
        parse.add_argument("--save_sample", action='store_true', help="Saves a sample jsonl", default=False)
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-c", "ConceptNet","-o", "create_artiset","-v", variant, \
                                 "-out","data/artiset_samples/ConceptNet/sample.jsonl","--save_sample"])

        if not os.path.exists("data/artiset_samples/ConceptNet"):
            os.mkdir('data/artiset_samples/ConceptNet')

        artiset = ConceptNet(args)

        # reducing data size to a sample:
        artiset._config['test_dev_size'] = [0, 0]
        artiset._config['max_number_of_examples'] = 20

        artiset.build_artificial_dataset(args)
