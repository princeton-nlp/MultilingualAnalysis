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


def align_labels(supervised, synthetic, task):
    threshold1 = 5
    threshold2 = 5
    # If the lengths are too different, ignore
    if abs(len(supervised["tokens"]) - len(synthetic["tokens"])) > threshold1:
        return None

    task_tag = '{}_tags'.format(task)

    default_tag = 'O' if task == 'ner' else 'X'

    # Initialize synthetic tokens to an empty list
    synthetic[task_tag] = []

    # Count the number of words that don't appear in the source sentence
    count_dont_appear = 0

    for i in range(len(synthetic["tokens"])):
        if synthetic["tokens"][i] not in supervised["tokens"]:
            synthetic[task_tag].append(default_tag)
            count_dont_appear += 1
        else:
            synthetic[task_tag].append(supervised[task_tag][supervised["tokens"].index(synthetic["tokens"][i])])

    if count_dont_appear > threshold2:
        return None

    return synthetic

def convert_to_document_ner(args):
    """
    JSON files need to follow these guidelines: https://huggingface.co/docs/datasets/loading_datasets.html#json-files
    """
    # Store lines in the file
    lines = open(args.galactic_file, 'r', encoding='utf8').readlines()

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

    # Ignored count
    ignored = 0

    # Also store it as a JSON
    json_file = os.path.join(file_dir, '{}_{}.json'.format('synthetic', original_file_name.split('.')[0]))
    f = open(json_file, 'w')
    for i, idx in enumerate(supervised_indices):
        temp_dict = {   
                        'tokens': synthetic[i].strip().split(),\
                        '{}_tags'.format(args.task): supervised_lines[idx]['{}_tags'.format(args.task)],\
                    }
        temp_dict = align_labels(supervised_lines[idx], temp_dict, args.task)
        if temp_dict is None:
            ignored += 1
        else:
            f.write(json.dumps(temp_dict)+'\n')
    f.close()

    # Print statistics
    print("Language: {} Task: {}".format(args.galactic_file, args.task))
    print("Ignored: {}%\n".format(ignored / len(supervised_indices) * 100))


def convert_conllu_to_document(args):
    # Check the task type
    if args.task == 'ner':
        convert_to_document_ner(args)
    elif args.task == 'pos':
        convert_to_document_ner(args)
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