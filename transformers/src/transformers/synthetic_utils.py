"""
Class and function definitions for word-based modifications
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import concatenate_datasets
from copy import deepcopy
import random

def create_modified_dataset(data_args, map_function, datasets):
    # # Create new dataset using map function
    # modified_dataset = datasets.map(
    #     map_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers
    # )

    if type(datasets) is dict:
        modified_dataset = {}
        for key in datasets.keys():
            modified_dataset[key] = datasets[key].map(
                map_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers
            )
    else:
        modified_dataset = datasets.map(
            map_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers
        )
    
    # Step 3: Check if a modified dataset needs to be ADDED or if it should be REPLACED
    if data_args.word_modification == 'add':
        # Check if there are multiple datasets or if it's a single dataset
        if 'keys' in dir(datasets):
            # Concatenate the two datasets
            combined_dataset = {}

            for key in datasets.keys():
                combined_dataset[key] = concatenate_datasets([datasets[key], modified_dataset[key]])
            
            return combined_dataset
        else:
            return concatenate_datasets([datasets, modified_dataset])

    elif data_args.word_modification == 'replace':
        # Check if there are multiple datasets or if it's a single dataset
        if 'keys' in dir(datasets):
            replaced_dataset = {}

            for key in modified_dataset.keys():
                replaced_dataset[key] = modified_dataset[key]
                
            return replaced_dataset
        else:
            return modified_dataset


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

    # Step 3: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)
        

def modify_inputs_words(data_args, training_args, datasets, task_name, tokenizer=None):
    # Get the sampling range for modifying the words
    sampling_range = [int(i) for i in data_args.modify_words_range.strip().split('-')]

    # Make sure the upper bound is lesser than the tokenizer length
    sampling_range[1] = min(sampling_range[1], len(tokenizer))

    # SEP, CLS, or PAD tokens
    sep_cls_pad = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token), tokenizer.convert_tokens_to_ids(tokenizer.pad_token)]

    # Step 1: Create map function for modification
    def map_function(examples):
        for j in range(len(examples['input_ids'])):
            # examples['input_ids'][j] = [examples['input_ids'][j][i] for i in range(len(examples['input_ids'][j])) if np.random.binomial(data_args.modify_words_probability) == 0 else np.random.randint(low=sampling_range[0], high=sampling_range[1])]
            examples['input_ids'][j] = [examples['input_ids'][j][i] if (np.random.binomial(1, data_args.modify_words_probability) == 0 or (not sampling_range[0] <= examples['input_ids'][j][i] <= sampling_range[1]) or examples['input_ids'][j][i] in sep_cls_pad) else np.random.randint(low=sampling_range[0], high=sampling_range[1]) for i in range(len(examples['input_ids'][j]))]
        return examples

    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)


def modify_inputs_invert_qa(data_args, training_args, datasets, task_name, tokenizer=None, negative_label=None):        
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # If the task is QA, then call a different function

    # TODO: </s> index is hard coded here. Pull it from the tokenizer instead.
    # TODO: pad_index is hard coded here. Pull it from the tokenizer instead.
    
    def map_function(examples):
        def reverse_list(s):
            s.reverse()
            return s
        
        def reverse_substr(sent_indices):
            temp_sent_indices = deepcopy(sent_indices)
            start_idx = 0
            current_idx = 0

            # Create a list with numbers from 0 to len(sent_indices)
            number_indices = list(range(len(sent_indices)))
            temp_number_indices = list(range(len(sent_indices)))

            sentence_length = len(sent_indices)

            # Indices to consider for [SEP] and [CLS]
            if tokenizer:
                sep_cls = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
            else:
                # If tokenizer is not passed, then use the default RoBERTa tokenizer tokens
                sep_cls = [0, 2]

            for i in range(sentence_length):
                if (sent_indices[i] in sep_cls) or (i == (sentence_length - 1)):
                    # flip sentence on start_idx, i
                    if i > start_idx:
                        if (i == (sentence_length - 1)) and (not (sent_indices[i] in sep_cls)):
                            sent_indices[start_idx: i+1] = reverse_list(temp_sent_indices[start_idx: i+1])
                            number_indices[start_idx: i+1] = reverse_list(temp_number_indices[start_idx: i+1])
                        else:
                            sent_indices[start_idx: i] = reverse_list(temp_sent_indices[start_idx: i])
                            number_indices[start_idx: i] = reverse_list(temp_number_indices[start_idx: i])
                    start_idx = i+1
            return sent_indices, number_indices

        for j in range(len(examples['input_ids'])):
            example_length = len(examples['input_ids'][j])
            modified_input_ids, number_indices = reverse_substr(examples['input_ids'][j])

            # Since it's a QA task, make modifications to other keys before `input_ids`
            # Train set
            if 'start_positions' in examples:
                temp = examples['start_positions'][j]
                examples['start_positions'][j] = number_indices[examples['end_positions'][j]]
                examples['end_positions'][j] = number_indices[temp]

            # Validation set
            if 'offset_mapping' in examples:
                examples['offset_mapping'][j] = [examples['offset_mapping'][j][idx] for idx in number_indices]

            # Modify the inputs
            examples['input_ids'][j] = [modified_input_ids[i] for i in range(example_length)]


        return examples

    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)


def modify_inputs_invert(data_args, training_args, datasets, task_name, tokenizer=None, negative_label=None):
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # If the task is QA, then call a different function
    if task_name == 'qa':
        return modify_inputs_invert_qa(data_args, training_args, datasets, task_name, tokenizer, negative_label)

    # TODO: </s> index is hard coded here. Pull it from the tokenizer instead.
    # TODO: pad_index is hard coded here. Pull it from the tokenizer instead.
    
    def map_function(examples):
        def reverse_list(s):
            s.reverse()
            return s
        # Reverse function for NER/POS labels
        def reverse_substr_ner_pos(sent_indices):
            temp_sent_indices = deepcopy(sent_indices)

            sentence_length = len(sent_indices)

            sent_indices[1:-1] = reverse_list(temp_sent_indices[1:-1])

            """
            # Now, if the labels were ['O', 'B-PER', 'I-PER'], they are modified to ['I-PER', 'B-PER', 'O']
            # Change it to ['B-PER', 'I-PER', 'O']
            # negative_label is the label index corresponding to 'O'
            if negative_label and task_name == 'ner':
                negative_labels = [negative_label]
                temp_sent_indices = deepcopy(sent_indices)
                start_idx = -1

                for i in range(sentence_length):
                    if sent_indices[i] not in negative_labels:
                        # Check if this is the first occurrence of an entity tag
                        if start_idx < 0:
                            start_idx = i
                            # If this is the last token in the sentence, then we don't have reverse it
                        elif  start_idx >= 0 and (i == (sentence_length - 1)):
                            # If it's the last token of the sentence and it is not 'O', then flip
                            sent_indices[start_idx: i+1] = reverse_list(temp_sent_indices[start_idx: i+1])
                    else:
                        if start_idx > 0:
                            # Flip the labels
                            sent_indices[start_idx: i] = reverse_list(temp_sent_indices[start_idx: i])
                            start_idx = -1
            """
                
            return sent_indices
        
        def reverse_substr(sent_indices):
            temp_sent_indices = deepcopy(sent_indices)
            start_idx = 0
            current_idx = 0

            sentence_length = len(sent_indices)

            # Indices to consider for [SEP] and [CLS]
            if tokenizer:
                sep_cls = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
            else:
                # If tokenizer is not passed, then use the default RoBERTa tokenizer tokens
                sep_cls = [0, 2]

            for i in range(sentence_length):
                if (sent_indices[i] in sep_cls) or (i == (sentence_length - 1)):
                    # flip sentence on start_idx, i
                    if i > start_idx:
                        if (i == (sentence_length - 1)) and (not (sent_indices[i] in sep_cls)):
                            sent_indices[start_idx: i+1] = reverse_list(temp_sent_indices[start_idx: i+1])
                        else:
                            sent_indices[start_idx: i] = reverse_list(temp_sent_indices[start_idx: i])
                    start_idx = i+1
            return sent_indices

        for j in range(len(examples['input_ids'])):
            example_length = len(examples['input_ids'][j])
            modified_examples = reverse_substr(examples['input_ids'][j])
            examples['input_ids'][j] = [modified_examples[i] for i in range(example_length)]
            # If it's a token classification task, flip the labels too
            if task_name in ['ner', 'pos']:
                modified_labels = reverse_substr_ner_pos(examples['labels'][j])
                examples['labels'][j] = [modified_labels[i] for i in range(example_length)]
        return examples

    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)       

def modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer):
    # Should we modify special tokens? That is contained in boolean data_args.shift_special
    if data_args.shift_special:
        special_tokens = [tokenizer.pad_token_id]
    else:
        special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

    # Vocabulary size
    vocab_size = tokenizer.vocab_size

    # If we are replacing only a fraction of the words, create a list
    # We use the same variable data_args.modify_words_probability here
    if data_args.one_to_one_file is not None:
        dont_modify = np.load(open(data_args.one_to_one_file, 'rb'))

        # Step 1: Create map function for modification
        def map_function(examples):
            for j in range(len(examples['input_ids'])):
                examples['input_ids'][j] = [examples['input_ids'][j][i] if (examples['input_ids'][j][i] in special_tokens or examples['input_ids'][j][i] in dont_modify) else (examples['input_ids'][j][i] + vocab_size)  for i in range(len(examples['input_ids'][j]))]
            return examples
    else:
        # Step 1: Create map function for modification
        def map_function(examples):
            for j in range(len(examples['input_ids'])):
                examples['input_ids'][j] = [examples['input_ids'][j][i] if (examples['input_ids'][j][i] in special_tokens) else (examples['input_ids'][j][i] + vocab_size)  for i in range(len(examples['input_ids'][j]))]
            return examples


    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)


def modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, tokenizer=None, negative_label=None):
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # This modification doesn't work for qa
    assert task_name != 'qa', "Permutation doesn't work for QA."
    
    def map_function(examples):
        def permute_list(sent, labels=None):
            if labels is not None:
                sent, labels = zip(*random.sample(list(zip(sent, labels)), len(sent)))
                return sent, labels
            else:
                sent = random.sample(sent, len(sent))
                return sent
        
        def permute_substr(sent_indices, sent_labels=None):
            temp_sent_indices = deepcopy(sent_indices)
            temp_sent_labels = deepcopy(sent_labels) if sent_labels is not None else None
            start_idx = 0
            current_idx = 0

            sentence_length = len(sent_indices)

            # Indices to consider for [SEP] and [CLS]
            if tokenizer:
                sep_cls = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
            else:
                # If tokenizer is not passed, then use the default RoBERTa tokenizer tokens
                sep_cls = [0, 2]

            for i in range(sentence_length):
                if (sent_indices[i] in sep_cls) or (i == (sentence_length - 1)):
                    # flip sentence on start_idx, i
                    if i > start_idx:
                        if (i == (sentence_length - 1)) and (not (sent_indices[i] in sep_cls)):
                            if sent_labels is not None:
                                sent_indices[start_idx: i+1], sent_labels[start_idx: i+1] = permute_list(temp_sent_indices[start_idx: i+1], labels=temp_sent_labels[start_idx: i+1])
                            else:
                                sent_indices[start_idx: i+1] = permute_list(temp_sent_indices[start_idx: i+1])
                        else:
                            if sent_labels is not None:
                                sent_indices[start_idx: i], sent_labels[start_idx: i] = permute_list(temp_sent_indices[start_idx: i], labels=temp_sent_labels[start_idx: i])
                            else:
                                sent_indices[start_idx: i] = permute_list(temp_sent_indices[start_idx: i])
                    start_idx = i+1
            if sent_labels is not None:
                return sent_indices, sent_labels
            else:
                return sent_indices

        for j in range(len(examples['input_ids'])):
            example_length = len(examples['input_ids'][j])
            if task_name in ['ner', 'pos']:
                # If it's a token classification task, flip the labels too
                modified_examples, modified_labels = permute_substr(examples['input_ids'][j], sent_labels=examples['labels'][j])
                examples['input_ids'][j] = [modified_examples[i] for i in range(example_length)]
                examples['labels'][j] = [modified_labels[i] for i in range(example_length)]
            else:
                modified_examples = permute_substr(examples['input_ids'][j])
                examples['input_ids'][j] = [modified_examples[i] for i in range(example_length)]

        return examples

    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)


def modify_inputs_synthetic(data_args, training_args, datasets, task_name=None, task_type='mlm', tokenizer=None):
    if task_type in ['glue', 'xnli', 'ner', 'pos', 'qa', 'tatoeba']:
        data_args.preprocessing_num_workers = None
    
    # If multiple word modifications are being performed, then handle them separately
    if data_args.one_to_one_mapping and data_args.invert_word_order:
        original_datasets = deepcopy(datasets)
        original_word_modification = data_args.word_modification
        data_args.word_modification = 'replace'
        datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer)
        datasets = modify_inputs_invert(data_args, training_args, datasets, task_name, tokenizer)

        if original_word_modification == 'replace':
            return datasets
        elif original_word_modification == 'add':
            if 'keys' in dir(datasets):
                # Concatenate the two datasets
                combined_dataset = {}

                for key in datasets.keys():
                    combined_dataset[key] = concatenate_datasets([original_datasets[key], datasets[key]])
                
                return combined_dataset

    # If multiple word modifications are being performed, then handle them separately
    if data_args.one_to_one_mapping and data_args.permute_words:
        original_datasets = deepcopy(datasets)
        original_word_modification = data_args.word_modification
        data_args.word_modification = 'replace'
        datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer)
        datasets = modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, tokenizer)

        if original_word_modification == 'replace':
            return datasets
        elif original_word_modification == 'add':
            if 'keys' in dir(datasets):
                # Concatenate the two datasets
                combined_dataset = {}

                for key in datasets.keys():
                    combined_dataset[key] = concatenate_datasets([original_datasets[key], datasets[key]])
                
                return combined_dataset

    # If we need to sample only a part of the dataset, handle it separately
    if 'target_dataset_ratio' in dir(data_args) and data_args.target_dataset_ratio is not None:
        original_datasets = deepcopy(datasets)
        original_word_modification = data_args.word_modification
        data_args.word_modification = 'replace'

    if data_args.permute_vocabulary:
        datasets = modify_inputs_permute(data_args, training_args, datasets, task_name)
    if data_args.modify_words:
        datasets = modify_inputs_words(data_args, training_args, datasets, task_name, tokenizer)
    if data_args.invert_word_order:
        datasets = modify_inputs_invert(data_args, training_args, datasets, task_name, tokenizer)
    if data_args.one_to_one_mapping:
        datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer)
    if data_args.permute_words:
        datasets = modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, tokenizer)

    # If we need to sample only a part of the dataset, handle it separately
    if 'target_dataset_ratio' in dir(data_args) and data_args.target_dataset_ratio is not None:
        # Subsample the original dataset
        for key in datasets.keys():
            if key == 'train':
                select_indices = random.sample(range(len(datasets[key])), int(data_args.target_dataset_ratio * len(datasets[key])))
                datasets[key] = datasets[key].select(select_indices)
        
        # Combine with original datasets
        if original_word_modification == 'replace':
            return datasets
        elif original_word_modification == 'add':
            if 'keys' in dir(datasets):
                # Concatenate the two datasets
                combined_dataset = {}

                for key in datasets.keys():
                    combined_dataset[key] = concatenate_datasets([original_datasets[key], datasets[key]])
                
                return combined_dataset   

    return datasets

def modify_config(data_args, training_args, config):
    if data_args.one_to_one_mapping:
        config.vocab_size = config.vocab_size * 2
        return config
    else:
        return config