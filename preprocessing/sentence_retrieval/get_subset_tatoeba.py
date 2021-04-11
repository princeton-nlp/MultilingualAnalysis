"""
From a monolingual or parallel corpus, get a subset of 1000 examples.
Make sure there are no repetitions either in the source or target language.
"""

import argparse
import os
import json

def get_subset_tatoeba(args):
    tatoeba_lines = open(args.filename, 'r').readlines()
    sep = '\t'

    split_lines = []
    for line in tatoeba_lines:
        line = line.strip().split(sep)
        split_lines.append(line)

    if args.language2:
        # If the second language exists
        # Save all the lines in json format
        tatoeba_file = os.path.join(args.save_dir, '{}_{}.json'.format(args.language1, args.language2))
        f = open(tatoeba_file, 'w')

        source_set = set()
        target_set = set()

        # Store a dictionary for source and target language and throw out
        # sentences which are repetitions (either in source or target language)
        count = 0
        for line in split_lines:
            
            if len(split_lines) > 2 * args.truncate:
                if line[0] in source_set or line[2] in target_set:
                    continue
                else:
                    source_set.add(line[0])
                    target_set.add(line[2])
                    count += 1
            
            temp_dict = {'sentence1': line[1], 'sentence2': line[3]}
            f.write(json.dumps(temp_dict)+'\n')

            if count >= args.truncate:
                break
        f.close()
    else:
        # Monolingual (for word modifications)
        # Save all the lines in json format
        tatoeba_file = os.path.join(args.save_dir, '{}.json'.format(args.language1))
        f = open(tatoeba_file, 'w')

        source_set = set()

        count = 0
        for line in split_lines:
            
            if len(split_lines) > 2 * args.truncate:
                if line[0] in source_set:
                    continue
                else:
                    source_set.add(line[0])
                    count += 1
                        
            temp_dict = {'sentence1': line[1]}
            f.write(json.dumps(temp_dict)+'\n')

            if count >= args.truncate:
                break
            
        f.close()

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--filename", type=str, required=True, help="")
    parser.add_argument("--save_dir", type=str, required=True, help="")
    parser.add_argument("--language1", type=str, required=True, help="")
    parser.add_argument("--language2", type=str, default=None, help="")
    parser.add_argument("--truncate", type=int, default=1000, help="")

    args = parser.parse_args()

    get_subset_tatoeba(args)

if __name__ == '__main__':
    main()