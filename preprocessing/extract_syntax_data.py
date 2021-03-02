"""
From a GD penn tree file, extract both the English and the modified English sentences
"""

# File to preprocess
file_name = '/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/galactic/treebanks-V1.0/treebanks/GD_English/en~hi@N~hi@V/en~hi@N~hi@V-gd-train.conllu'

# English and modified file name
english_file = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/syntax_modifications/conllu_english.txt'
modified_file = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/syntax_modifications/conllu_modified_english.txt'

# Open files to write
eng = open(english_file, 'w')
mod = open(modified_file, 'w')

# Read the file
lines = open(file_name, 'r').readlines()
english_lines = []
modified_lines = []

# Iterate over the lines
for line in lines:
    temp = line.strip().split(' ', 2)
    if temp[0] == '#' and temp[1] == 'sentence-tokens-src:':
        english_lines.append('{}\n'.format(temp[2]))
    elif temp[0] == '#' and temp[1] == 'sentence-tokens:':
        modified_lines.append('{}\n'.format(temp[2]))

# Write to the files
eng.writelines(english_lines)
mod.writelines(modified_lines)