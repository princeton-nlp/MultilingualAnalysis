"""
This file loads in the XNLI dataset and converts it to sentences which can be passed to the dependency parser.
"""

import argparse
from tqdm import tqdm
import os
import csv
import json
import newlinejson as nlj


def convert_xnli_to_sentences(args):
    # Open JSON file
    lines = []
    with nlj.open(args.xnli_file) as src:
        for line in src:
            lines.append(line)

    # Store the sentences
    sentences = []

    for line in lines:
        sentences.append(line['sentence1']+'\n')
        sentences.append(line['sentence2']+'\n')

    # Save the file
    write_file_name = os.path.join(args.save_dir, '{}_{}'.format('flattened', os.path.split(args.xnli_file)[1]))
    open(write_file_name, 'w').writelines(sentences)


def convert_dataset_to_sentences(args):
    # Check the task type
    if args.task == 'xnli':
        convert_xnli_to_sentences(args)
    else:
        raise('This script works only for XNLI')

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--cache_dir", default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models/stanza', type=str, help="Directory to store models and files.")
    parser.add_argument("--xnli_file", required=True, type=str, help="Path to file containing the corpus.")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to directory where file will be saved.")
    parser.add_argument("--task", default='xnli', type=str, help="mlm/mnli/xnli/..../")
    parser.add_argument("--language", default='en', type=str, help="Language code if using a multilingual dataset like XNLI")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_dataset_to_sentences(args)

if __name__ == '__main__':
    main()