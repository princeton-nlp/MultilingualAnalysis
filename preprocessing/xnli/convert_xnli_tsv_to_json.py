import argparse
from tqdm import tqdm
import os
import pandas
import json


def convert_xnli_to_json(args):
    if args.mode == 'train':
        xnli_file = os.path.join(args.xnli_dir, 'multinli.train.{}.tsv'.format(args.language))

        # Header
        sep = '\t'
        header = sep.join(['premise', 'hypo', 'label'])+'\n'

        # Directory to write in
        xnli_json_file = os.path.join(args.xnli_save_dir, '{}_{}.json'.format(args.mode, args.language))

        # Get the training dataset lines
        training_lines = open(xnli_file, 'r').readlines()[1:]
        lines_to_save = []

        for line in training_lines:
            line_split = line.strip().split('\t')
            lines_to_save.append([line_split[0], line_split[1], line_split[-1]])

        # Save all the lines in json format
        f = open(xnli_json_file, 'w')
        for line in lines_to_save:
            temp_dict = {'sentence1': line[0], 'sentence2': line[1], 'label': line[2]}
            f.write(json.dumps(temp_dict)+'\n')
        f.close()
    elif args.mode == 'dev':
        xnli_file = os.path.join(args.xnli_dir, 'xnli.dev.tsv'.format(args.language))

        # Header
        sep = '\t'
        header = ['language' 'gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'promptID', 'pairID', 'genre', 'label1', 'label2', 'label3', 'label4', 'label5', 'sentence1_tokenized', 'sentence2_tokenized', 'match']

        # Directory to write in
        xnli_json_file = os.path.join(args.xnli_save_dir, '{}_{}.json'.format(args.mode, args.language))

        # Get the training dataset lines
        training_lines = open(xnli_file, 'r').readlines()[1:]
        lines_to_save = []

        for line in training_lines:
            line_split = line.strip().split('\t')
            if line_split[0] == args.language:
                lines_to_save.append([line_split[6], line_split[7], line_split[1]])

        # Save all the lines in json format
        f = open(xnli_json_file, 'w')
        for line in lines_to_save:
            temp_dict = {'sentence1': line[0], 'sentence2': line[1], 'label': line[2]}
            f.write(json.dumps(temp_dict)+'\n')
        f.close()        

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--mode", type=str, required=True, help="train/dev")
    parser.add_argument("--xnli_dir", type=str, default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/multilingual_nlu/XNLI/XNLI-MT-1.0/multinli/', help="")
    parser.add_argument("--xnli_save_dir", type=str, required=True, help="")
    parser.add_argument("--language", type=str, required=True, help="")

    args = parser.parse_args()

    convert_xnli_to_json(args)

if __name__ == '__main__':
    main()