import argparse

from LeapOfThought.artiset_factory import ArtisetFactory

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-c","--artiset_name", type=str, help="The name of the artiset class and config to use")
    parse.add_argument("-o","--operation", type=str, help="The task stage to run")
    parse.add_argument("-out","--output_file", type=str, help="")
    parse.add_argument("-e", "--experiment_version", type=str, help="mini version, appended as a suffix to the artiset file name", default="")
    parse.add_argument("-v", "--variant", type=str, help="", default="")
    parse.add_argument("-s", "--split", type=str, help="Build a specific split", default=None)
    parse.add_argument("-fs", "--split_by_field", type=str,
                       help="dev+test vs train split so that the specified field would be disjoint", default="")
    parse.add_argument("--cuda_device", type=int, help="", default=-1)
    parse.add_argument("--copy_from", type=str, help="For create new artiset, the chllenge to copy from", default=-1)
    parse.add_argument("--artiset_module", type=str, help="For create new artiset, the target artiset path", default='')
    parse.add_argument("-p", "--n_processes", type=int, help="For artisets with multi process", default=1)
    parse.add_argument("--config_path", type=str, help="Artisets config file", default="config.json")
    parse.add_argument("--s3_subdirs", type=str, help="list separated by ,")
    parse.add_argument("--s3_prefix", type=str, help="list separated by ,")
    parse.add_argument("--sample_sizes", type=str, help="list separated by ,")
    parse.add_argument("--dev_sample_sizes", type=str, help="list separated by ,")
    parse.add_argument("--save_sample", action='store_true', help="Saves a sample jsonl", default=False)
    parse.add_argument("--incorrect_beliefs_file", type=str, help="Jsonl file with incorrect belief edges.", default=None)
    args = parse.parse_args()

    if args.operation == 'create_new_artiset':
        ArtisetFactory().create_new_artiset(args.artiset_name, args.artiset_module, args.copy_from, args)
    elif args.operation == 'combine_artisets':
        # Note, in this procedure artiset_name and sample_size are both lists separated by ","
        ArtisetFactory().combine_artisets(args.s3_subdirs, args.s3_prefix, args.sample_sizes, args.dev_sample_sizes, args.output_file, args)
    else:
        if args.operation == 'build_artificial_dataset':
            if len(args.variant) > 0 and len(args.variant.split(','))>1:
                variants = args.variant.split(',')
            else:
                variants = [args.variant]

            if len(args.experiment_version) > 0 and len(args.experiment_version.split(',')) > 1:
                experiment_versions = args.experiment_version.split(',')
            else:
                experiment_versions = [args.experiment_version]

            for variant in variants:
                for experiment_version in experiment_versions:
                    args.variant = variant
                    args.experiment_version = experiment_version
                    logger.info(f'building {variant} variant {experiment_version} experiment_version')
                    artiset = ArtisetFactory().get_artiset(args.artiset_name, args)
                    artiset.build_artificial_dataset(args)

        elif args.operation == 'preprocess_artiset':
            artiset = ArtisetFactory().get_artiset(args.artiset_name, args)
            artiset.preprocess_artiset(args)
        else:
            logger.error('Operation not supported')

if __name__ == '__main__':
    main()
