"""
This script loads in a file which is the output of galactic dependencies dataset.
It converts it both to a synthetic language corpus.
"""

import argparse
from tqdm import tqdm
import os
import pandas
import json
import newlinejson as nlj


def convert_to_document_tatoeba(args):
    """
    JSON files need to follow these guidelines: https://huggingface.co/docs/datasets/loading_datasets.html#json-files
    """
    # Store lines in the file
    lines = open(args.galactic_file, 'r').readlines()

    # Store both in the monolingual and synthetic language corpus
    monolingual = []
    synthetic = []

    # Parse the files
    for line in lines:
        # If line starts with `# sentence-tokens-src:` then it is monolingual corpus
        if line.startswith('# sentence-tokens-src:'):
            start_string = '# sentence-tokens-src:'
            monolingual.append(line[len(start_string):])
        elif line.startswith('# sentence-tokens:'):
            start_string = '# sentence-tokens:'
            synthetic.append(line[len(start_string):])

    # Locate file directory
    file_dir, original_file_name = os.path.split(args.galactic_file)

    # Also store it as a JSON
    json_file = os.path.join(file_dir, '{}_{}.json'.format('synthetic', original_file_name.split('.')[0]))
    f = open(json_file, 'w')

    for line in synthetic:
        temp_dict = {'sentence1': line.strip()}
        f.write(json.dumps(temp_dict)+'\n')        
    f.close()


def convert_conllu_to_document(args):
    # Check the task type
    convert_to_document_tatoeba(args)

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--galactic_file", type=str, default='en', help="File with galactic dependencies output")
    parser.add_argument("--cache_dir", default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models/stanza', type=str, help="Directory to store models and files.")
    parser.add_argument("--task", default='mlm', type=str, help="mlm/mnli/xnli/..../")
    parser.add_argument("--verbose", action="store_true", help="Verbose or now")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_conllu_to_document(args)

if __name__ == '__main__':
    main()