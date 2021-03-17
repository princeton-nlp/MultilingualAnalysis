"""
This script loads in a file which is the output of galactic dependencies dataset.
It converts it both to a monolingual corpus a synthetic language corpus.
"""

import argparse
from tqdm import tqdm
import os

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

    # Store the monolingual file
    synthetic_file = os.path.join(file_dir, '{}_{}'.format('synthetic', original_file_name))
    f = open(synthetic_file, 'w')
    f.writelines(monolingual + synthetic)
    f.close()




def convert_conllu_to_document(args):
    # Check the task type
    if args.task == 'mlm':
        convert_to_document_mlm(args)
    else:
        raise('No support for this task type: {}'.format(args.task))

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