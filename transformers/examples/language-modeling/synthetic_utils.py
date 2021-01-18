"""
Class and function definitions for word-based modifications
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import concatenate_datasets

def modify_inputs_permute(data_args, training_args, datasets):
    # Step 1: Load the vocab mapping
    # Function for modifying string json to integer json
    # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
    def jsonKV2int(x):
        if isinstance(x, dict):
                return {int(k):(int(v) if isinstance(v, str) else v) for k,v in x.items()}
        return x
    
    # Load the the vocabulary file
    if data_args.permute_vocabulary:
        with open(data_args.vocab_permutation_file, 'r') as fp:
            vocab_mapping = json.load(fp, object_hook=jsonKV2int)

    # Step 2: Check if a modified dataset needs to be ADDED or if it should be REPLACED
    if data_args.vocab_modification == 'add':
        # Map function for datasets.map
        def map_function(examples):
            for j in range(len(examples['input_ids'])):
                examples['input_ids'][j] = [vocab_mapping[examples['input_ids'][j][i]] for i in range(len(examples['input_ids'][j]))]
            return examples

        # Create new dataset using map function
        modified_dataset = datasets.map(
            map_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers
        )

        # Concatenate the two datasets
        combined_dataset = {}
        if training_args.do_train:
            combined_dataset['train'] = concatenate_datasets([datasets['train'], modified_dataset['train']])

        if training_args.do_eval:
            combined_dataset['validation'] = concatenate_datasets([datasets['validation'], modified_dataset['validation']])

        return combined_dataset

        

# class WordBasedModifications():
#     def __init__(self, data_args):
#         self.data_args = data_args

#         # Function for modifying string json to integer json
#         # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
#         def jsonKV2int(x):
#             if isinstance(x, dict):
#                     return {int(k):(int(v) if isinstance(v, str) else v) for k,v in x.items()}
#             return x
        
#         # Load the the vocabulary file
#         if self.data_args.permute_vocabulary:
#             with open(self.data_args.vocab_permutation_file, 'r') as fp:
#                 self.vocab_mapping = json.load(fp, object_hook=jsonKV2int)

#     def modify_inputs_permute(self, inputs):
#         # Information about inputs:
#         # inputs['input_ids'].device is cpu
#         # inputs['input_ids'] is torch.Tensor

#         # TODO: Optimize this code. Currently using for loops

#         # Check if all the inputs need to be modified
#         if self.data_args.vocab_modification == 'random':
#             # With a 50% probability, just return the original inputs
#             if np.random.uniform() < 0.5:
#                 return inputs


#         # for i in range(inputs['input_ids'].shape[0]):
#         #     for j in range(inputs['input_ids'].shape[1]):
#         #         inputs['input_ids'][i,j] = self.vocab_mapping[inputs['input_ids'][i,j].item()]
#         #         # if inputs['labels'][i,j] >= 0:
#         #         #     inputs['labels'][i,j] = self.vocab_mapping[inputs['labels'][i,j].item()]
#         for i in range(len(inputs['input_ids'])):
#             inputs['input_ids'][i] = self.vocab_mapping[inputs['input_ids'][i]]               

#         return inputs

#     def modify_inputs_permute_all(self, train_dataset):
#         """
#         Modify all the inputs in the dataset
#         """
        
#         length_of_dataset = len(train_dataset)
        
#         # Print statement
#         print("Word-based transformation: Permuting the vocabulary")

#         # Loop over all the sentences
#         for i in tqdm(range(len(train_dataset))):
#             modified_inputs = self.modify_inputs_permute(train_dataset[i])
#             train_dataset[i]['input_ids'] = modified_inputs['input_ids']
#             # train_dataset[i]['labels'] = modified_inputs['labels']

#         return train_dataset