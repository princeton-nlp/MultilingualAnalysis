"""
This file loads in a dataset (like MNLI, XNLI, SST-2) and converts it to sentences which can be passed to the dependency parser.
"""

import argparse
from tqdm import tqdm
import os
import csv


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def convert_xnli_to_sentences(args):
    # Convert XNLI for a single language and store it in MNLI format
    
    # Process the train file first #########################################
    train_file = os.path.join(args.data_dir, 'XNLI-MT-1.0/multinli/', 'multinli.train.{}.tsv'.format(args.language))

    # NOTE: Try to save it in the same format as MNLI. It's okay if dummy variables are added to columns that don't exist
    lines = read_tsv(train_file)

    # Store the sentences
    sentences = []

    # Count the number of sentences which have more than `args.truncate` words
    count_truncate = 0
    total_count = 0

    # Sentence indices in the dataset
    sent_1_index = 0
    sent_2_index = 1

    for idx, line in enumerate(lines):
        # Ignore if it is the first line
        if idx == 0:
            continue

        if len(line[sent_1_index].strip().split()) >= args.truncate:
            count_truncate += 1
            temp = ' '.join(line[sent_1_index].strip().split()[:args.truncate])
            sentences.append(temp+'\n')
        else:
            sentences.append(line[sent_1_index]+'\n')

        if len(line[sent_2_index].strip().split()) >= args.truncate:
            count_truncate += 1
            temp = ' '.join(line[sent_2_index].strip().split()[:args.truncate])
            sentences.append(temp+'\n')
        else:
            sentences.append(line[sent_2_index]+'\n')

        total_count += 2

    # Print statistics
    print("% of lines above {} words in XNLI train: {:.2f}%, total: {}, and count above {}: {}".format(args.truncate, count_truncate/total_count*100, total_count, args.truncate, count_truncate))

    # Save the file
    write_file_name = os.path.join(args.save_dir, '{}_{}_{}_train.txt'.format('flattened', 'xnli', args.language))
    open(write_file_name, 'w').writelines(sentences)

    # Process the dev file #########################################
    dev_file = os.path.join(args.data_dir, 'XNLI-1.0/', 'xnli.dev.tsv')

    # NOTE: Try to save it in the same format as MNLI. It's okay if dummy variables are added to columns that don't exist
    lines = read_tsv(dev_file)

    # Store the sentences
    sentences = []

    # Count the number of sentences which have more than `args.truncate` words
    count_truncate = 0
    total_count = 0

    # Sentence indices in the dataset
    sent_1_index = 6
    sent_2_index = 7
    lang_index = 0

    for idx, line in enumerate(lines):
        # Ignore if it is the first line
        if idx == 0:
            continue

        # Check if it's the correct language
        if line[lang_index] != args.language:
            continue

        if len(line[sent_1_index].strip().split()) >= args.truncate:
            count_truncate += 1
            temp = ' '.join(line[sent_1_index].strip().split()[:args.truncate])
            sentences.append(temp+'\n')
        else:
            sentences.append(line[sent_1_index]+'\n')

        if len(line[sent_2_index].strip().split()) >= args.truncate:
            count_truncate += 1
            temp = ' '.join(line[sent_2_index].strip().split()[:args.truncate])
            sentences.append(temp+'\n')
        else:
            sentences.append(line[sent_2_index]+'\n')

        total_count += 2

    # Print statistics
    print("% of lines above {} words in XNLI dev: {:.2f}%, total: {}, and count above {}: {}".format(args.truncate, count_truncate/total_count*100, total_count, args.truncate, count_truncate))

    # Save the file
    write_file_name = os.path.join(args.save_dir, '{}_{}_{}_dev.txt'.format('flattened', 'xnli', args.language))
    open(write_file_name, 'w').writelines(sentences)    

def convert_mnli_to_sentences(args):
    # List all the files that need to be converted
    # file_list = ['train.tsv', 'dev_matched.tsv', 'dev_mismatched.tsv']
    file_list = ['small_dev_matched.tsv']
    file_list = [os.path.join(args.data_dir, file_name) for file_name in file_list]
    print("MNLI test files are not being converted.")

    # Load each file and split on tabs
    for file_name in file_list:
        lines = read_tsv(file_name)

        # Store the sentences, one after another
        # HACK: If the line number of the sentence is x, it is the first sentence if x%2==0
        sentences = []

        # Count the number of sentences which have more than `args.truncate` words
        count_truncate = 0
        total_count = 0        

        for idx, line in enumerate(lines):
            # Ignore if it is the first line
            if idx == 0:
                continue

            if len(line[8].strip().split()) >= args.truncate:
                count_truncate += 1
                temp = ' '.join(line[8].strip().split()[:args.truncate])
                sentences.append(temp+'\n')
            else:
                sentences.append(line[8]+'\n')

            if len(line[9].strip().split()) >= args.truncate:
                count_truncate += 1
                temp = ' '.join(line[9].strip().split()[:args.truncate])
                sentences.append(temp+'\n')
            else:
                sentences.append(line[9]+'\n')

            total_count += 2

        # Print statistics
        print("% of lines above {} words in {}: {:.2f}%, total: {}, and count above {}: {}".format(args.truncate, os.path.split(file_name)[-1], count_truncate/total_count*100, total_count, args.truncate, count_truncate))
        
        # Save the file
        write_file_name = os.path.join(args.save_dir, '{}_{}.txt'.format('flattened', os.path.split(file_name)[-1].split('.')[0]))
        open(write_file_name, 'w').writelines(sentences)

def convert_dataset_to_sentences(args):
    # Check the task type
    if args.task == 'mnli':
        convert_mnli_to_sentences(args)
    elif args.task == 'xnli':
        convert_xnli_to_sentences(args)
    else:
        raise('No support for this task type: {}'.format(args.task))

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--cache_dir", default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models/stanza', type=str, help="Directory to store models and files.")
    parser.add_argument("--data_dir", required=True, type=str, help="Path to file containing the corpus.")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to directory where file will be saved.")
    parser.add_argument("--task", default='mlm', type=str, help="mlm/mnli/xnli/..../")
    parser.add_argument("--truncate", default=1000000, type=int, help="Truncate sentences which are greater than this length. This is essential because galactic dependencies code can't handle very long sentences.")
    parser.add_argument("--verbose", action="store_true", help="Verbose or now")
    parser.add_argument("--language", default='en', type=str, help="Language code if using a multilingual dataset like XNLI")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_dataset_to_sentences(args)

if __name__ == '__main__':
    main()