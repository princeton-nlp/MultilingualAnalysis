"""
This file loads in a dataset and saves the dependency parse for each sentence.
It uses StanfordNLP's Stanza library
"""

import argparse
from tqdm import tqdm
import os
from copy import deepcopy

import stanza
from stanza.utils.conll import CoNLL


def check_non_projectivity(doc_dep):
    """
    Return False if the tree is non-projective
    0 denotes the root, and not -1
    Indexing is 1-based
    """

    # In some languages some words are split like (du = de le).
    # We ignore such congregations while checking non-projectivity.
    doc_dep = [doc_dep[i] for i in range(len(doc_dep)) if '-' not in doc_dep[i][0]]
    
    # Index for HEAD is 6 and for current index is 0
    HEAD = 6
    CURRENT = 0

    # Flag to store if tree is non-projective
    flag = True

    for i in range(len(doc_dep)):
        for j in range(len(doc_dep)):
            head1 = int(doc_dep[i][HEAD])
            head2 = int(doc_dep[j][HEAD])
            child1 = int(doc_dep[i][CURRENT])
            child2 = int(doc_dep[j][CURRENT])

            # Ignore if either of the heads are root
            if head1 <= 0 or head2 <= 0:
                continue

            # If they have the same heads, then ignore because they won't cross
            if head1 == head2:
                continue

            # Check if the arcs cross
            if (child1 > head1 and head1 != head2):
                if ((child1 > head2 and child1 < child2 and head1 < head2) or (child1 < head2 and child1 > child2 and head1 < child2)):
                    flag = False
                    break
            if (child1 < head1 and head1 != head2):
                if ((head1 > head2 and head1 < child2 and child1 < head2) or (head1 < head2 and head1 > child2 and child1 < child2)):
                    flag = False
                    break

    return flag


def check_if_works_galactic(doc_dep):
    """
    There are four "InvalidTreeException conditions that need to be satisfied from here:
    https://github.com/gdtreebank/gdtreebank/blob/d83bbaa92fc0ad9db0c254182d21ce6924b5aa91/src/main/java/grammar/NaryTree.java#L334
    
    1) If any of the sentences in an instance have a word with >6 dependents
    then we ignore that instance entirely. This is so that the galactic
    dependencies code doesn't ignore it.

    2) if (Constant.filterPuncts > 0 && punctHead): Since the constant is always 0,
    this constraint is always satisfied

    Assume you care only about words up until the first period in each sentence
    """

    # Care only about the first sentence
    doc_dep = doc_dep[0]

    # Build a dictionary to see how many dependents each word has
    num_dependents = {}

    # Iterate over all the words
    for word in doc_dep:
        
        # Update the count for its parent
        if word[6] not in num_dependents:
            num_dependents[word[6]] = 0
        num_dependents[word[6]] += 1

    # Iterate over the constructed dictionary and check if there are >6 dependents
    flag = True
    for key, value in num_dependents.items():
        if value > 6:
            flag = False

    # # NOTE: Might have to do this for NER
    # # In some languages some words are split like (du = de le). We ignore such examples
    # for i in range(len(doc_dep)):
    #     if '-' in doc_dep[i][0]:
    #         flag = False

    # Check for non-projectivity of dependency parse of the sentence
    # Do this only if flag is true
    if flag:
        flag = flag and check_non_projectivity(doc_dep)

    return flag


def convert_to_conllu_tatoeba(args):
    """
    A flattened corpus of MNLI is passed.
    The format is:
    Instance 1
    Instance 2
    Instance 3
    ......
    ......

    If any of the sentences in an instance have a word with >6 dependents
    then we ignore that instance entirely. This is so that the galactic
    dependencies code doesn't ignore it.
    """
    # Load the file
    sentences = open(args.data, 'r').readlines()

    # Instantiate a model
    stanza.download(args.language, model_dir=args.cache_dir, verbose=args.verbose)

    # Construct a pipeline
    print("Building a pipeline for {}".format(args.language))
    en_nlp = stanza.Pipeline(args.language, dir=args.cache_dir)

    # List to store all the dependency parse information
    # Store it as [[dep for sent 1, dep for sent 2], [dep for sent 1, dep for sent 2], ...]
    dep_parse = []
    instances_selected = []
    ignored = 0

    # Tag all sentences and store in memory
    # `en_doc` is empty list if it is a sentence with only spaces/tabs/newlines
    for idx, sentence in tqdm(enumerate(sentences), desc='Parse sentences'):
        # Ignore if it's an empty sentence
        if not sentence:
            continue
        
        # Parse the sentence
        en_doc = en_nlp(sentence)
        doc_dict = en_doc.to_dict()
        doc_dep = CoNLL.convert_dict(doc_dict)

        # Check if the sentence works for the galactic dependencies code
        flag = check_if_works_galactic(doc_dep)

        if flag:
            dep_parse.append(doc_dep)
            instances_selected.append(idx)
        else:
            ignored += 1

    # Print how many instances were ignored
    print("Ignored {}% of instances".format(ignored / len(sentences) * 100))

    # After getting the valid sentences, store them in a file
    # Open file to write
    _, original_file_name = os.path.split(args.data)
    file_name = os.path.join(args.save_dir, 'dep_{}'.format(original_file_name))
    dep_file = open(file_name, 'w')

    for sentence in tqdm(dep_parse, desc='Save sentences'):
        # Check if it's an empty list
        if not sentence:
            continue

        # Word and its information
        # NOTE: Care only about the words up until the first period. Else they are processed as different sentences.
        for word in sentence[0]:
            word_and_info = '\t'.join(word)
            word_and_info = word_and_info + '\n'
            dep_file.write(word_and_info)

        # Print a new line
        dep_file.write('\n')

    # Close the file
    dep_file.close()

    # After storing the dependency information, also store what indices were selected
    file_name = os.path.join(args.save_dir, 'selected_indices_{}'.format(original_file_name))
    select_file = open(file_name, 'w')

    # Iterate over the indices
    for idx in instances_selected:
        select_file.write(str(idx)+'\n')
    
    # Close the file
    select_file.close()

def convert_document_to_conllu(args):
    # Check the task type
    if args.task == 'tatoeba':
        convert_to_conllu_tatoeba(args)        
    else:
        raise('No support for this task type: {}'.format(args.task))


def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--language", type=str, default='en', help="Language code of the dependency parser.")
    parser.add_argument("--cache_dir", default='/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models/stanza', type=str, help="Directory to store models and files.")
    parser.add_argument("--data", required=True, type=str, help="Path to file containing the corpus.")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to directory where file will be saved.")
    parser.add_argument("--task", default='mlm', type=str, help="mlm/mnli/xnli/..../")
    parser.add_argument("--verbose", action="store_true", help="Verbose or now")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_document_to_conllu(args)

if __name__ == '__main__':
    main()