"""
There is no need for iterative normalization because the real and synthetic language are isomorphic

Example command: python calculate_embedding_mse.py --model1 ~/bucket/model_outputs/en/one_to_one_mapping_100_500K/mlm/ --model2 ~/bucket/model_outputs/en/one_to_one_mapping_100_500K/mlm/ --train_fraction 0.05 --valid_fraction 0.9
"""

import argparse
import numpy as np
from transformers import AutoModelForMaskedLM, AutoConfig

def get_embeddings(args):
    # Instantiate the two models
    config = AutoConfig.from_pretrained(args.model1)
    model1 = AutoModelForMaskedLM.from_pretrained(args.model1, config=config)
    config = AutoConfig.from_pretrained(args.model2)
    model2 = AutoModelForMaskedLM.from_pretrained(args.model2, config=config)
    embeddings1 = model1.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy()
    embeddings2 = model2.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy()

    embeddings = [embeddings1, embeddings2]

    # Use first half of the embeddings for the first model and second half for the second
    vocab_size = int(embeddings[0].shape[0] / 2)

    # Use the first half of the embeddings for the first model, and second half for the second
    embeddings[0] = embeddings[0][:vocab_size]
    embeddings[1] = embeddings[1][vocab_size:2 * vocab_size]

    return embeddings


def compute_embeddings_mse(embeddings):
    # Print the shapes of the embedding matrices
    print("Shape of original embedding is: {} and derived embedding is: {}".format(embeddings[0].shape, embeddings[0].shape))

    # Compute mean euclidean distance
    mean_distance = np.mean(np.sqrt(np.sum(np.square(embeddings[0] - embeddings[1]), axis=1)))
    print("Mean euclidean distance: {:.2f}".format(float(mean_distance)))



def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--model1", type=str, required=True, help="")
    parser.add_argument("--model2", default=None, type=str, help="")
    parser.add_argument("--train_fraction", type=float, default=0.1)
    parser.add_argument("--valid_fraction", type=float, default=0.4)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    args.model2 = args.model1

    embeddings = get_embeddings(args)

    # Compute MSE between embeddings
    compute_embeddings_mse(embeddings)


if __name__ == '__main__':
    main()