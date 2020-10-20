import argparse
import os
import glob
import json
from tqdm import tqdm


def get_data():
    os.system('wget https://dl.fbaipublicfiles.com/LAMA/data.zip')
    os.system('unzip data.zip')
    os.system('rm data.zip')


def collect_data():
    data = {}
    for f_name in tqdm(glob.glob('data/Google_RE/*.jsonl')):
        with open(f_name, 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        key_name = f_name.split('/')[-1]
        data[key_name] = []
        for line in lines:
            row = json.loads(line)
            data[key_name].append((row['sub_label'], row['obj_label']))

    return data

def clean_data_files():
    os.system('rm -r data/')


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-o", "--output_file", type=str, help="Output file of the trex data")
    args = parse.parse_args()

    # get_data()
    data = collect_data()

    with open(args.output_file, 'w') as f:
        json.dump(data, f)

    # clean_data_files()


if __name__ == '__main__':
    main()
