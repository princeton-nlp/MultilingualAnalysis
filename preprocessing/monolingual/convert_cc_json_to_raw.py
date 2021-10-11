"""
Convert common crawl json files to raw text.
Data downloaded from: https://huggingface.co/datasets/allenai/c4/tree/main
"""

import newlinejson as nlj
import argparse
import os


def convert_json_to_raw_text(args):
    # Open the file
    lines = []
    
    with nlj.open(args.file) as src:
        for line in src:
            lines.append(line)

    # Write all the lines to a new file
    file_dir, original_file_name = os.path.split(args.file)
    new_file_name = os.path.join(file_dir, '{}.txt'.format(original_file_name.rsplit('.', 1)[0]))

    f = open(new_file_name, 'w')
    for line in lines:
        f.write(line['text'] + '\n')

    f.close()

    


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--file", type=str, default='temp.txt', help="")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_json_to_raw_text(args)

if __name__ == '__main__':
    main()