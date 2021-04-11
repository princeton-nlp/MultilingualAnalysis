"""
Usage:
`python one_to_one_mapping.py ---vocab_size 50265 --random_seed 42 --fraction_to_replace 0.5`
"""

import numpy as np
import argparse
import os

def word_indices_to_ignore(args):
    np.random.seed = args.random_seed
    dont_modify = np.random.choice(args.vocab_size, int(args.vocab_size * (1 - args.fraction_to_replace)))

    # Save the array
    file_name = os.path.join('configuration_files',  'one_to_one_mapping_random_{}_fraction_{}.npy'.format(args.vocab_size, int(100 * args.fraction_to_replace)))

    with open(file_name, 'wb') as f:
        np.save(f, dont_modify)


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size of the tokenizer")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for creating a permutation")
    parser.add_argument("--fraction_to_replace", type=float, required=True, help="Random seed for creating a permutation")

    args = parser.parse_args()

    word_indices_to_ignore(args)

if __name__ == '__main__':
    main()