from indictrans import Transliterator
import argparse
import newlinejson as nlj
import os
import json

"""
Usage
python transliterate_finetune.py --file ../../../data/xnli/hi/dev_hi.json --save_dir ../../../data/transliteration/ --task xnli
"""


def transliterate_xnli(args):
    trn = Transliterator(source='hin', target='eng', build_lookup=True)

    # Load the dataset
    # Open JSON file
    lines = []
    with nlj.open(args.file) as src:
        for line in src:
            lines.append(line)

    for line in lines:
        line['sentence1'] = trn.transform(line['sentence1'])
        line['sentence2'] = trn.transform(line['sentence2'])

    # Transliterate each text piece
    # Also store it as a JSON
    _, file_name = os.path.split(args.file)
    json_file = os.path.join(args.save_dir, '{}_{}'.format(args.task, file_name))
    f = open(json_file, 'w')
    for line in lines:
        f.write(json.dumps(line)+'\n')        
    f.close()


def transliterate_ner(args):
    trn = Transliterator(source='hin', target='eng', build_lookup=True)

    # Load the dataset
    # Open JSON file
    lines = []
    with nlj.open(args.file) as src:
        for line in src:
            lines.append(line)

    for line in lines:
        new_sentence = []
        for token in line['tokens']:
            new_sentence.append(trn.transform(token))
        line['tokens'] = new_sentence

    # Transliterate each text piece
    # Also store it as a JSON
    _, file_name = os.path.split(args.file)
    json_file = os.path.join(args.save_dir, '{}_{}'.format(args.task, file_name))
    f = open(json_file, 'w')
    for line in lines:
        f.write(json.dumps(line)+'\n')        
    f.close()


def transliterate_tatoeba(args):
    trn = Transliterator(source='hin', target='eng', build_lookup=True)

    # Load the dataset
    # Open JSON file
    lines = []
    with nlj.open(args.file) as src:
        for line in src:
            lines.append(line)

    for line in lines:
        line['sentence2'] = trn.transform(line['sentence2'])

    # Transliterate each text piece
    # Also store it as a JSON
    _, file_name = os.path.split(args.file)
    json_file = os.path.join(args.save_dir, '{}_{}'.format(args.task, file_name))
    f = open(json_file, 'w')
    for line in lines:
        f.write(json.dumps(line)+'\n')        
    f.close()    


def transliterate_dataset(args):
    # Check the task type
    if args.task == 'xnli':
        transliterate_xnli(args)
    elif args.task == 'ner':
        transliterate_ner(args)
    elif args.task == 'pos':
        transliterate_ner(args)
    elif args.task == 'tatoeba':
        transliterate_tatoeba(args)                
    else:
        raise('No support for this task type: {}'.format(args.task))

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--cache_dir", default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models/stanza', type=str, help="Directory to store models and files.")
    parser.add_argument("--file", required=True, type=str, help="Path to file containing the corpus.")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to directory where file will be saved.")
    parser.add_argument("--task", default='mlm', type=str, help="mlm/xnli/ner/pos/tatoeba/..../")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    transliterate_dataset(args)

if __name__ == '__main__':
    main()