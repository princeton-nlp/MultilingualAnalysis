"""
This file loads in a dataset and saves the dependency parse for each sentence.
It uses StanfordNLP's Stanza library
"""

import argparse
from tqdm import tqdm
import os

import stanza
from stanza.utils.conll import CoNLL

def convert_to_conllu_mlm(args):
    # Load the file
    sentences = open(args.data, 'r').readlines()

    # Instantiate a model
    stanza.download(args.language, model_dir=args.cache_dir, verbose=args.verbose)

    # Construct a pipeline
    print("Building a pipeline for {}".format(args.language))
    en_nlp = stanza.Pipeline(args.language, dir=args.cache_dir)

    # # # sentence = "The Apprentice Boys ' parade is an annual celebration by unionists of the relief of the Siege of Derry in 1689 , which began when thirteen young apprentice boys shut the city 's gates against the army of King James . At that time the parade was held on 12 August each year . Participants from across Northern Ireland and Britain marched along the city walls above the Bogside , and were often openly hostile to the residents . On 30 July 1969 the Derry Citizens Defence Association ( DCDA ) was formed to try to preserve peace during the period of the parade , and to defend the Bogside and Creggan in the event of an attack . The chairman was Seán Keenan , an Irish Republican Army ( IRA ) veteran ; the vice @-@ chairman was Paddy Doherty , a popular local man sometimes known as \" Paddy Bogside \" and the secretary was Johnnie White , another leading republican and leader of the James Connolly Republican Club . Street committees were formed under the overall command of the DCDA and barricades were built on the night of 11 August . The parade took place as planned on 12 August . As it passed through Waterloo Place , on the edge of the Bogside , hostilities began between supporters and opponents of the parade . Fighting between the two groups continued for two hours , then the police joined in . They charged up William Street against the Bogsiders , followed by the ' Paisleyites ' . They were met with a hail of stones and petrol bombs . The ensuing battle became known as the Battle of the Bogside . Late in the evening , having been driven back repeatedly , the police fired canisters of CS gas into the crowd . Youths on the roof of a high @-@ rise block of flats on Rossville Street threw petrol bombs down on the police . Walkie @-@ talkies were used to maintain contact between different areas of fighting and DCDA headquarters in Paddy Doherty 's house in Westland Street , and first aid stations were operating , staffed by doctors , nurses and volunteers . Women and girls made milk @-@ bottle crates of petrol bombs for supply to the youths in the front line and \" Radio Free Derry \" broadcast to the fighters and their families . On the third day of fighting , 14 August , the Northern Ireland Government mobilised the Ulster Special Constabulary ( B @-@ Specials ) , a force greatly feared by nationalists in Derry and elsewhere . Before they engaged , however , British troops were deployed at the scene , carrying automatic rifles and sub @-@ machine guns . The RUC and B @-@ Specials withdrew , and the troops took up positions outside the barricaded area . "
    # sentence = "Some random sentence ."

    # # en_doc = en_nlp(sentence)

    # List to store all the dependency parse information
    dep_parse = []

    # Tag all sentences and store in memory
    # `en_doc` is empty list if it is a sentence with only spaces/tabs/newlines
    for sentence in tqdm(sentences, desc='Parse sentences'):
        try:
            en_doc = en_nlp(sentence)
            doc_dict = en_doc.to_dict()
            doc_dep = CoNLL.convert_dict(doc_dict)
            dep_parse.append(doc_dep)
        except:
            continue

    # Convert the tagged sentences to CONLLU format and store it
    # Example format is here: https://github.com/gdtreebank/gdtreebank/blob/master/toy/sample.conllu

    # Open file to write
    _, original_file_name = os.path.split(args.data)
    file_name = os.path.join(args.save_dir, 'dep_{}'.format(original_file_name))
    dep_file = open(file_name, 'w')

    for sentences in tqdm(dep_parse, desc='Save sentences'):
        # Check if it's an empty list
        if not sentences:
            continue
        
        for sentence in sentences:
            # Word and its information
            for word in sentence:
                word_and_info = '\t'.join(word)
                word_and_info = word_and_info + '\n'
                dep_file.write(word_and_info)

            # Print a new line
            dep_file.write('\n')

    # Close the file
    dep_file.close()


def check_non_projectivity(doc_dep):
    """
    Return False if the tree is non-projective
    0 denotes the root, and not -1
    Indexing is 1-based
    """
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

    # Check for non-projectivity of dependency parse of the sentence
    # Do this only if flag is true
    if flag:
        flag = flag and check_non_projectivity(doc_dep)

    return flag


def convert_to_conllu_mnli(args):
    """
    A flattened corpus of MNLI is passed.
    The format is:
    Instance 1 sentence 1
    Instance 1 sentence 2
    Instance 2 sentence 1
    Instance 2 sentence 2
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
    dep_parse_for_instance = []
    instances_selected = []

    # Tag all sentences and store in memory
    # `en_doc` is empty list if it is a sentence with only spaces/tabs/newlines
    for idx, sentence in tqdm(enumerate(sentences), desc='Parse sentences'):
        # Ignore if it's an empty sentence
        if not sentence:
            continue

        # Check if we are parsing the first sentence in an instance or the second
        if idx % 2 == 0:
            # If flag is True, then add it to the dep_parse, else ignore
            dep_parse_for_instance = []
            flag = True
        
        # Parse the sentence
        en_doc = en_nlp(sentence)
        doc_dict = en_doc.to_dict()
        doc_dep = CoNLL.convert_dict(doc_dict)

        # Check if the sentence works for the galactic dependencies code
        flag = flag and check_if_works_galactic(doc_dep) 
        dep_parse_for_instance.append(doc_dep)

        # If we have reached the second sentence, then append the doc
        if idx % 2 == 1:
            if flag:
                dep_parse.append(dep_parse_for_instance)
                instances_selected.append(int((idx-1)/2))

    # After getting the valid sentences, store them in a file
    # Open file to write
    _, original_file_name = os.path.split(args.data)
    file_name = os.path.join(args.save_dir, 'dep_{}'.format(original_file_name))
    dep_file = open(file_name, 'w')

    for sentences in tqdm(dep_parse, desc='Save sentences'):
        # Check if it's an empty list
        if not sentences:
            continue
        
        # Go over both the instances
        for sentence in sentences:
            # Word and its information
            # NOTE: Care only about the words up until the first period
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
    if args.task == 'mlm':
        convert_to_conllu_mlm(args)
    elif args.task == 'mnli':
        convert_to_conllu_mnli(args)
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