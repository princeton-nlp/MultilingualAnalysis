"""
File for creating tokenizer files
Code borrowed from: https://huggingface.co/blog/how-to-train
"""


# Libraries for tokenizer
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import argparse
import json
import os


def create_tokenizer(args):

    # Directory for storing
    directory = args.store_files

    # Train the tokenizer
    # paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]
    paths = [args.file]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    # tokenizer.save(args.store_files)
    tokenizer.save_model(args.store_files)

    tokenizer_config = {
        "max_len": 512
    }

    with open("{}/tokenizer_config.json".format(args.store_files), 'w') as fp:
        json.dump(tokenizer_config, fp)


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--file", type=str, required=True, help="File name to use for creating the tokenizer")
    parser.add_argument("--store_files", type=str, required=True, help="Where to store the tokenizer files")
    parser.add_argument("--vocab_size", default=52000, type=int, help="How many WordPiece tokens to use")

    args = parser.parse_args()

    if not os.path.exists(args.store_files):
        os.makedirs(args.store_files)    

    create_tokenizer(args)

if __name__ == '__main__':
    main()