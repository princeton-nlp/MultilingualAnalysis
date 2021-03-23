"""
This script loads in a file which is the output of galactic dependencies dataset.
It converts it both to a monolingual corpus a synthetic language corpus.
"""

import argparse
from tqdm import tqdm
import os
import pandas
import json


def convert_to_document_mnli(args):
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
    supervised_lines = open(args.supervised_dataset, 'r').readlines()
    header = supervised_lines[0]
    supervised_lines = supervised_lines[1:]
    supervised_indices = [int(idx) for idx in open(args.index_selector, 'r').readlines()]

    # Locate file directory
    file_dir, original_file_name = os.path.split(args.galactic_file)

    # Store the monolingual file
    mono_file = os.path.join(file_dir, '{}_{}.tsv'.format('mono', original_file_name.split('.')[0]))
    f = open(mono_file, 'w')

    # Header
    sep = '\t'
    header = sep.join(['sentence1', 'sentence2', 'label'])+'\n'

    # Write the header
    f.write(header)
    for i, idx in enumerate(supervised_indices):
        line = []
        # First sentence
        line.append(monolingual[2 * i].strip())
        # Second sentence
        line.append(monolingual[2 * i + 1].strip())
        # Label
        line.append(supervised_lines[idx].strip().split()[-1])

        # Combine the line
        line = sep.join(line)+'\n'

        f.write(line)
    f.close()

    # Also store it as JSON
    json_file = os.path.join(file_dir, '{}_{}.json'.format('mono', original_file_name.split('.')[0]))
    f = open(json_file, 'w')
    json_str = pandas.read_csv(mono_file, delimiter='\t').to_json(orient='records')
    json_list = json.loads(json_str)
    for sent in json_list:
        f.write(json.dumps(sent)+'\n')
    f.close()

    # Store the synthetic file
    synthetic_file = os.path.join(file_dir, '{}_{}.tsv'.format('synthetic', original_file_name.split('.')[0]))
    f = open(synthetic_file, 'w')

    # Write the header
    f.write(header)
    for i, idx in enumerate(supervised_indices):
        line = []
        # First sentence
        line.append(synthetic[2 * i].strip())
        # Second sentence
        line.append(synthetic[2 * i + 1].strip())
        # Label
        line.append(supervised_lines[idx].strip().split()[-1])

        line = sep.join(line)
        f.write(line+'\n')
    f.close()

    # Also store it as a JSON
    json_file = os.path.join(file_dir, '{}_{}.json'.format('synthetic', original_file_name.split('.')[0]))
    f = open(json_file, 'w')
    for i, idx in enumerate(supervised_indices):
        temp_dict = {'sentence1': synthetic[2 * i].strip(), 'sentence2': synthetic[2 * i + 1].strip(), 'label': supervised_lines[idx].strip().split()[-1]}
        f.write(json.dumps(temp_dict)+'\n')        
    f.close()


def convert_to_document_mlm(args):
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
            monolingual.append(line[len(start_string):]+'\n')
        elif line.startswith('# sentence-tokens:'):
            start_string = '# sentence-tokens:'
            synthetic.append(line[len(start_string):]+'\n')

    # Locate file directory
    file_dir, original_file_name = os.path.split(args.galactic_file)

    # Store the monolingual file
    mono_file = os.path.join(file_dir, '{}_{}'.format('mono', original_file_name))
    f = open(mono_file, 'w')
    f.writelines(monolingual)
    f.close()

    # Store the synthetic file
    synthetic_file = os.path.join(file_dir, '{}_{}'.format('synthetic', original_file_name))
    f = open(synthetic_file, 'w')
    f.writelines(monolingual + synthetic)
    f.close()


def convert_conllu_to_document(args):
    # Check the task type
    if args.task == 'mlm':
        convert_to_document_mlm(args)
    elif args.task == 'mnli':
        convert_to_document_mnli(args)        
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