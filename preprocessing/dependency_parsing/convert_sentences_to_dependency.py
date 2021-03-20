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
    en_nlp = stanza.Pipeline('en', dir=args.cache_dir)

    # # # sentence = "The Apprentice Boys ' parade is an annual celebration by unionists of the relief of the Siege of Derry in 1689 , which began when thirteen young apprentice boys shut the city 's gates against the army of King James . At that time the parade was held on 12 August each year . Participants from across Northern Ireland and Britain marched along the city walls above the Bogside , and were often openly hostile to the residents . On 30 July 1969 the Derry Citizens Defence Association ( DCDA ) was formed to try to preserve peace during the period of the parade , and to defend the Bogside and Creggan in the event of an attack . The chairman was Seán Keenan , an Irish Republican Army ( IRA ) veteran ; the vice @-@ chairman was Paddy Doherty , a popular local man sometimes known as \" Paddy Bogside \" and the secretary was Johnnie White , another leading republican and leader of the James Connolly Republican Club . Street committees were formed under the overall command of the DCDA and barricades were built on the night of 11 August . The parade took place as planned on 12 August . As it passed through Waterloo Place , on the edge of the Bogside , hostilities began between supporters and opponents of the parade . Fighting between the two groups continued for two hours , then the police joined in . They charged up William Street against the Bogsiders , followed by the ' Paisleyites ' . They were met with a hail of stones and petrol bombs . The ensuing battle became known as the Battle of the Bogside . Late in the evening , having been driven back repeatedly , the police fired canisters of CS gas into the crowd . Youths on the roof of a high @-@ rise block of flats on Rossville Street threw petrol bombs down on the police . Walkie @-@ talkies were used to maintain contact between different areas of fighting and DCDA headquarters in Paddy Doherty 's house in Westland Street , and first aid stations were operating , staffed by doctors , nurses and volunteers . Women and girls made milk @-@ bottle crates of petrol bombs for supply to the youths in the front line and \" Radio Free Derry \" broadcast to the fighters and their families . On the third day of fighting , 14 August , the Northern Ireland Government mobilised the Ulster Special Constabulary ( B @-@ Specials ) , a force greatly feared by nationalists in Derry and elsewhere . Before they engaged , however , British troops were deployed at the scene , carrying automatic rifles and sub @-@ machine guns . The RUC and B @-@ Specials withdrew , and the troops took up positions outside the barricaded area . "
    # sentence = "Some random sentence ."

    # # en_doc = en_nlp(sentence)

    # List to store all the dependency parse information
    dep_parse = []

    # Tag all sentences and store in memory
    # `en_doc` is empty list if it is a sentence with only spaces/tabs/newlines
    for sentence in tqdm(sentences, desc='Parse sentences'):
        en_doc = en_nlp(sentence)
        doc_dict = en_doc.to_dict()
        doc_dep = CoNLL.convert_dict(doc_dict)
        dep_parse.append(doc_dep)

    # Convert the tagged sentences to CONLLU format and store it
    # Example format is here: https://github.com/gdtreebank/gdtreebank/blob/master/toy/sample.conllu

    # Open file to write
    _, original_file_name = os.path.split(args.data)
    file_name = os.path.join(args.save_dir, 'dep_{}'.format(original_file_name))
    dep_file = open(file_name, 'w')

    for sentence in tqdm(dep_parse, desc='Save sentences'):
        # Check if it's an empty list
        if not sentence:
            continue

        # Go one list level down. `sentence` is not a list of words and their dep_parse information
        sentence = sentence[0]

        # Word and its information
        for word in sentence:
            word_and_info = '\t'.join(word)
            word_and_info = word_and_info + '\n'
            dep_file.write(word_and_info)

        # Print a new line
        dep_file.write('\n')

    # Close the file
    dep_file.close()


def convert_document_to_conllu(args):
    # Check the task type
    if args.task == 'mlm':
        convert_to_conllu_mlm(args)
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