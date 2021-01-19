"""
Class and function definitions for word-based modifications
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import concatenate_datasets

def modify_inputs_permute(data_args, training_args, datasets, task_name):
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
            
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # Step 2: Create a modified dataset
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
    
    # Step 3: Check if a modified dataset needs to be ADDED or if it should be REPLACED
    if data_args.word_modification == 'add':
        # Concatenate the two datasets
        combined_dataset = {}
        if training_args.do_train:
            combined_dataset['train'] = concatenate_datasets([datasets['train'], modified_dataset['train']])

        if training_args.do_eval:
            dict_key = "validation_matched" if task_name == "mnli" else "validation"
            combined_dataset[dict_key] = concatenate_datasets([datasets[dict_key], modified_dataset[dict_key]])
        return combined_dataset

    elif data_args.word_modification == 'replace':
        replaced_dataset = {}
        if training_args.do_train:
            replaced_dataset['train'] = modified_dataset['train']

        if training_args.do_eval:
            dict_key = "validation_matched" if task_name == "mnli" else "validation"
            replaced_dataset[dict_key] = modified_dataset[dict_key]
        return replaced_dataset
        

def modify_inputs_words(data_args, training_args, datasets, task_name):
    # Get the sampling range for modifying the words
    sampling_range = [int(i) for i in data_args.modify_words_range.strip().split('-')]

    # Step 1: Create map function for modification
    def map_function(examples):
        for j in range(len(examples['input_ids'])):
            # examples['input_ids'][j] = [examples['input_ids'][j][i] for i in range(len(examples['input_ids'][j])) if np.random.binomial(data_args.modify_words_probability) == 0 else np.random.randint(low=sampling_range[0], high=sampling_range[1])]
            examples['input_ids'][j] = [examples['input_ids'][j][i] if (np.random.binomial(1, data_args.modify_words_probability) == 0 or (not sampling_range[0] <= examples['input_ids'][j][i] <= sampling_range[1])) else np.random.randint(low=sampling_range[0], high=sampling_range[1]) for i in range(len(examples['input_ids'][j]))]
        return examples

    # Create new dataset using map function
    modified_dataset = datasets.map(
        map_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers
    )
    
    # Step 3: Check if a modified dataset needs to be ADDED or if it should be REPLACED
    if data_args.word_modification == 'add':
        # Concatenate the two datasets
        combined_dataset = {}
        if training_args.do_train:
            combined_dataset['train'] = concatenate_datasets([datasets['train'], modified_dataset['train']])

        if training_args.do_eval:
            dict_key = "validation_matched" if task_name == "mnli" else "validation"
            combined_dataset[dict_key] = concatenate_datasets([datasets[dict_key], modified_dataset[dict_key]])
        return combined_dataset

    elif data_args.word_modification == 'replace':
        replaced_dataset = {}
        if training_args.do_train:
            replaced_dataset['train'] = modified_dataset['train']

        if training_args.do_eval:
            dict_key = "validation_matched" if task_name == "mnli" else "validation"
            replaced_dataset[dict_key] = modified_dataset[dict_key]
        return replaced_dataset

def modify_inputs(data_args, training_args, datasets, task_name=None, task_type='mlm'):
    if task_type == 'glue':
        data_args.preprocessing_num_workers = None
    if data_args.permute_vocabulary:
        datasets = modify_inputs_permute(data_args, training_args, datasets, task_name)
    if data_args.modify_words:
        datasets = modify_inputs_words(data_args, training_args, datasets, task_name)

    return datasets