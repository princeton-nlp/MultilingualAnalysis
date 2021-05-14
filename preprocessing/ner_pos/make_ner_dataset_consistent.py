"""
Changes the labels to make them consistent.
For example, `I-ORG B-ORG O` is changed to `B-ORG I-ORG O`
"""

import argparse
from tqdm import tqdm
import os
import csv
import json
import newlinejson as nlj
import copy


def make_tags_consistent(tags):
    new_tags = copy.deepcopy(tags)
    
    flag = False
    prev_suffix = None
    
    for i in range(len(new_tags)):
        if 'I' not in tags[i] and 'B' not in tags[i]:
            flag = False
            prev_suffix = None
        elif not flag and 'I' in tags[i]:
            new_tags[i] = 'B-'+tags[i].split('-')[-1]
            flag = True
            prev_suffix = tags[i].split('-')[-1]
        elif not flag and 'B' in tags[i]:
            flag = True
            prev_suffix = tags[i].split('-')[-1]
        elif flag and 'B' in tags[i] and prev_suffix == tags[i].split('-')[-1]:
            new_tags[i] = 'I-'+tags[i].split('-')[-1]
        elif flag and 'B' in tags[i] and prev_suffix != tags[i].split('-')[-1]:
            prev_suffix = tags[i].split('-')[-1]
        elif flag and 'I' in tags[i] and prev_suffix != tags[i].split('-')[-1]:
            new_tags[i] = 'B-'+tags[i].split('-')[-1]
            prev_suffix = tags[i].split('-')[-1]            

    return new_tags


def make_consistent(args):
    # Open JSON file
    lines = []
    with nlj.open(args.file) as src:
        for line in src:
            lines.append(line)

    for line in lines:
        line['ner_tags'] = make_tags_consistent(line['ner_tags'])

    # Write the file
    

    # Store the sentences
    sentences = []

    for line in lines:
        sentences.append(' '.join(line['tokens'])+'\n')

    # Save the file
    save_dir, old_file_name = os.path.split(args.file)
    write_file_name = os.path.join(save_dir, 'consistent_{}'.format(old_file_name))

    f = open(write_file_name, 'w')
    for line in lines:
        f.write(json.dumps(line)+'\n')
    f.close()


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--file", required=True, type=str, help="Path to file containing the corpus.")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    make_consistent(args)

if __name__ == '__main__':
    main()