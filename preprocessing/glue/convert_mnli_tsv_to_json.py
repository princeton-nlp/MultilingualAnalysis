import argparse
from tqdm import tqdm
import os
import pandas
import json

mnli_file = '/n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/glue_data_new/MNLI/train.tsv'

# Header
sep = '\t'
header = sep.join(['sentence1', 'sentence2', 'label'])+'\n'

# Directory to write in
save_dir = '/n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/dependency_parse_data/english/MNLI/en_hi_N_hi_V'
mnli_json_file = os.path.join(save_dir, 'train.json')
mnli_tsv_file = os.path.join(save_dir, 'train.tsv')

# Get the training dataset lines
training_lines = open(mnli_file, 'r').readlines()[1:]
lines_to_save = []

for line in training_lines:
    line_split = line.strip().split('\t')
    lines_to_save.append([line_split[8], line_split[9], line_split[-1]])

# Save all the lines in tsv format
f = open(mnli_tsv_file, 'w')
f.write(header)
for line in lines_to_save:
    f.write('\t'.join(line)+'\n')
f.close()

# # Remove problematic lines
# problems = open(mnli_tsv_file, 'r').readlines()
# for idx, line in enumerate(problems):
#     if len(line.strip().split('\t')) != 3:
#         print("Error in index {}".format(idx))

# Convert the tsv to json

# Save all the lines in tsv format
f = open(mnli_json_file, 'w')
for line in lines_to_save:
    temp_dict = {'sentence1': line[0], 'sentence2': line[1], 'label': line[2]}
    f.write(json.dumps(temp_dict)+'\n')
f.close()

# json_str = pandas.read_csv(mnli_tsv_file, delimiter='\t').to_json(orient='records')
# json_list = json.loads(json_str)
# for sent in json_list:
#     f.write(json.dumps(sent)+'\n')
# f.close()