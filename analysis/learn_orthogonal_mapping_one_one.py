"""
There is no need for iterative normalization because the real and synthetic language are isomorphic

Example command: python learn_orthogonal_mapping_one_one.py --model1 ~/bucket/model_outputs/en/one_to_one_mapping_100_500K/mlm/ --model2 ~/bucket/model_outputs/en/one_to_one_mapping_100_500K/mlm/ --train_fraction 0.05 --valid_fraction 0.9
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


def get_orthogonal_mapping(embeddings, args, train):
    # Normalize both the embedding matrices
    embeddings[0] = embeddings[0]/np.linalg.norm(embeddings[0], axis=1, keepdims=True)
    embeddings[1] = embeddings[1]/np.linalg.norm(embeddings[1], axis=1, keepdims=True)

    # Singular value decomposition
    U, Sigma, V = np.linalg.svd(embeddings[1][train].T @ embeddings[0][train])

    # Construct the orthogonal mapping
    W = U @ V

    return W, embeddings


def test_mapping(embeddings, orthogonal, args, validation, train):
    ############## Validation Accuracy #################
    # Compute the similarity matrix for words in the validation set

    # Compute the n X m similarity matrix
    # similarity = (embeddings[0][validation] @ orthogonal.T) @ embeddings[1][validation].T
    similarity = (embeddings[0][validation] @ orthogonal.T) @ embeddings[1].T

    # Compute argmax for each row
    selected_indices = np.argmax(similarity, axis=1)

    # Compute the accuracy
    total_correct = 0
    # accuracy = np.sum(selected_indices == np.arange(selected_indices.shape[0])) / selected_indices.shape[0] * 100
    accuracy = np.sum(selected_indices == np.array(validation)) / selected_indices.shape[0] * 100

    # Print the accuracy
    print("Validation accuracy is: {}".format(accuracy))
    print("Support: {}".format(selected_indices.shape[0]))


    ####### Validation Accuracy without alignment ######
    # Compute the n X n similarity matrix
    # similarity = embeddings[0][validation] @ embeddings[1][validation].T
    similarity = embeddings[0][validation] @ embeddings[1].T

    # Compute argmax for each row
    selected_indices = np.argmax(similarity, axis=1)

    # Compute the accuracy
    # accuracy = np.sum(selected_indices == np.arange(selected_indices.shape[0])) / selected_indices.shape[0] * 100
    accuracy = np.sum(selected_indices == np.array(validation)) / selected_indices.shape[0] * 100

    # Print the accuracy
    print("Validation accuracy without alignment is: {}".format(accuracy))    


    ################# Train Accuracy ####################
    # Compute the similarity matrix for words in the validation set

    # Compute the n X n similarity matrix
    # similarity = (embeddings[0][train] @ orthogonal.T) @ embeddings[1][train].T
    similarity = (embeddings[0][train] @ orthogonal.T) @ embeddings[1].T

    # Compute argmax for each row
    selected_indices = np.argmax(similarity, axis=1)

    # Compute the accuracy
    # accuracy = np.sum(selected_indices == np.arange(selected_indices.shape[0])) / selected_indices.shape[0] * 100
    accuracy = np.sum(selected_indices == np.array(train)) / selected_indices.shape[0] * 100

    # Print the accuracy
    print("Train accuracy is: {}".format(accuracy))    


def get_train_validation_indices(embeddings, args):
    vocab_size = embeddings[0].shape[0]
    indices = list(range(vocab_size))
    np.random.shuffle(indices)
    train, validation = indices[:int(vocab_size * args.train_fraction)], indices[-int(vocab_size * args.valid_fraction):]

    return train, validation


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

    train, validation = get_train_validation_indices(embeddings, args)

    orthogonal, embeddings = get_orthogonal_mapping(embeddings, args, train)

    test_mapping(embeddings, orthogonal, args, validation, train)

if __name__ == '__main__':
    main()