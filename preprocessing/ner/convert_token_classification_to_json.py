import argparse
import json
import os


def convert_pos_to_json(args):
    # Get path to language directory
    language_dir = args.data_dir

    # Convert all the files to JSON
    for split in ['train', 'dev']:
        pos_lines = open(os.path.join(language_dir, split+'-'+args.language+'.tsv')).readlines()

        # Open file to write JSON
        f = open(os.path.join(language_dir, split+'-'+args.language+'.json'), 'w')

        json_dict = {'tokens': [], 'pos_tags': []}

        for idx, line in enumerate(pos_lines):
            if line.strip() == '':
                f.write(json.dumps(json_dict)+'\n')
                json_dict = {'tokens': [], 'pos_tags': []}
            else:
                line = line.strip().split('\t')
                json_dict['tokens'].append(line[0])
                json_dict['pos_tags'].append(line[1])
        
        # Close the file
        f.close()


def convert_ner_to_json(args):
    # Get path to language directory
    language_dir = os.path.join(args.data_dir, args.language)

    # Convert all the files to JSON
    for split in ['train', 'dev', 'test']:
        ner_lines = open(os.path.join(language_dir, split)).readlines()

        # Open file to write JSON
        f = open(os.path.join(language_dir, split+'.json'), 'w')

        json_dict = {'tokens': [], 'ner_tags': []}

        for line in ner_lines:
            if line.strip() == '':
                f.write(json.dumps(json_dict)+'\n')
                json_dict = {'tokens': [], 'ner_tags': []}
            else:
                line = line.split(':', 1)[1]
                line = line.strip().split('\t')
                json_dict['tokens'].append(line[0])
                json_dict['ner_tags'].append(line[1])
        
        # Close the file
        f.close()

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--data_dir", type=str, required=True, help="")
    parser.add_argument("--task_name", default='ner', type=str, help="ner/pos")
    parser.add_argument("--language", default='en', type=str, help="Language code")

    args = parser.parse_args()

    if args.task_name == 'ner':
        convert_ner_to_json(args)
    elif args.task_name == 'pos':
        convert_pos_to_json(args)


if __name__ == '__main__':
    main()