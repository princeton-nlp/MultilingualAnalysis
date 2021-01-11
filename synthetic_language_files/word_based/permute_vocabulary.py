"""
Usage:
`python permute_vocabulary.py --vocab_size 50265 --random_seed 42 --ignore_until 200`
`python permute_vocabulary.py --vocab_file ../../config/wiki_vocab_english/vocab.json --random_seed 42 --ignore_until 200`
"""

import argparse
import numpy as np
import json

def create_permutation(args):
    # Check if vocab file is provided
    if args.vocab_file:
        with open(args.vocab_file, 'r') as fp:
            vocabulary = json.load(fp)
        args.vocab_size = len(vocabulary)

    # Initialize the original and modified vocabulary
    original_vocabulary = np.array(range(args.vocab_size))
    modified_vocabulary = np.array(range(args.vocab_size))

    # Random seed
    np.random.seed(args.random_seed)

    # Randomly permute part of the array after ignore_until index
    modified_vocabulary[args.ignore_until:] = np.random.permutation(original_vocabulary[args.ignore_until:])

    # Create a mapping and store it as a json
    vocab_mapping = {}
    for i in range(len(original_vocabulary)):
        vocab_mapping[str(original_vocabulary[i])] = str(modified_vocabulary[i])

    if args.vocab_file:
        # Ensure that the <mask> token number is not permuted
        mask_index = str(vocabulary["<mask>"])

        # Get the inverted vocabulary mapping
        inverted_vocab_mapping = {v: k for k, v in vocab_mapping.items()}
        new_mask_index = inverted_vocab_mapping[(mask_index)]

        # Set the <mask> index correctly
        word_swapped_with_mask = vocab_mapping[mask_index]
        vocab_mapping[mask_index] = mask_index
        vocab_mapping[new_mask_index] = word_swapped_with_mask

    # Save the vocabulary file
    with open('configuration_files/permuted_vocab_seed_{}_size_{}.json'.format(args.random_seed, args.vocab_size), 'w') as fp:
        json.dump(vocab_mapping, fp)


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size of the tokenizer")
    parser.add_argument("--vocab_file", type=str, default=None, help="Random seed for creating a permutation")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for creating a permutation")
    parser.add_argument("--ignore_until", default=200, type=int, help="Ignore permutation until index ignore_until")

    args = parser.parse_args()

    create_permutation(args)

if __name__ == '__main__':
    main()