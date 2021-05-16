"""
Check how many words are split as a results of tokenization
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("../config/en/roberta_8/")
# tokenizer = AutoTokenizer.from_pretrained("../config/en/vocab_20000/")

# model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", cache_dir=cache_dir)

# Tokenize all the words in the file
sentence_file = '../../data/dependency_parse_data/en/valid.txt'
# sentence_file = '../../data/xnli/en/flattened_dev_en.json'
sentences = open(sentence_file, 'r').readlines()

# Keep count of the number of words which are split
count_total = 0
count_split = 0

# Loop over all the words
total_words = 0
total_tokens = 0
for sentence in sentences:
    if len(sentence.strip().split()) < 2:
        continue
    total_words += len(sentence.strip().split())

    total_tokens += (len(tokenizer.encode(sentence)) - 2)

# Print fraction of words split
print("Expansion ratio = {}".format(total_tokens / total_words))
print("Total words and tokens {} {}".format(total_words, total_tokens))