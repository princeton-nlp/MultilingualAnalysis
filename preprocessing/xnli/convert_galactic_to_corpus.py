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


def convert_to_document_xnli(args):
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

    # Extract lines from the supervised dataset
    # Open JSON file
    supervised_lines = []
    with nlj.open(args.supervised_dataset) as src:
        for line in src:
            supervised_lines.append(line)    
    supervised_indices = [int(idx) for idx in open(args.index_selector, 'r').readlines()]

    # Locate file directory
    file_dir, original_file_name = os.path.split(args.galactic_file)

    # Also store it as a JSON
    json_file = os.path.join(file_dir, '{}_{}.json'.format('synthetic', original_file_name.split('.')[0]))
    f = open(json_file, 'w')
    for i, idx in enumerate(supervised_indices):
        temp_dict = {'sentence1': synthetic[2 * i].strip(), 'sentence2': synthetic[2 * i + 1].strip(), 'label': supervised_lines[idx]['label']}
        f.write(json.dumps(temp_dict)+'\n')        
    f.close()


def convert_conllu_to_document(args):
    # Check the task type
    if args.task == 'xnli':
        convert_to_document_xnli(args)        
    else:
        raise('No support for this task type: {}'.format(args.task))

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--galactic_file", type=str, default='en', help="File with galactic dependencies output")
    parser.add_argument("--supervised_dataset", type=str, default=None, help="Path of the supervised dataset.")
    parser.add_argument("--index_selector", type=str, default=None, help="Used for supervised datasets. Used for selecting indices from the original dataset.")
    parser.add_argument("--cache_dir", default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models/stanza', type=str, help="Directory to store models and files.")
    parser.add_argument("--task", default='mlm', type=str, help="mlm/mnli/xnli/..../")
    parser.add_argument("--verbose", action="store_true", help="Verbose or now")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_conllu_to_document(args)

if __name__ == '__main__':
    main()