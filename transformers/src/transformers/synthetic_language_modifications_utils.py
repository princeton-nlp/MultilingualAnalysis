"""
Class and function definitions for word-based modifications
"""

import json
import torch
import numpy as np

class WordBasedModifications():
    def __init__(self, data_args):
        self.data_args = data_args

        # Function for modifying string json to integer json
        # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
        def jsonKV2int(x):
            if isinstance(x, dict):
                    return {int(k):(int(v) if isinstance(v, str) else v) for k,v in x.items()}
            return x
        
        # Load the the vocabulary file
        if self.data_args.permute_vocabulary:
            with open(self.data_args.vocab_permutation_file, 'r') as fp:
                self.vocab_mapping = json.load(fp, object_hook=jsonKV2int)

    def modify_inputs_permute(self, inputs):
        # Information about inputs:
        # inputs['input_ids'].device is cpu
        # inputs['input_ids'] is torch.Tensor

        # # Function for substituting indices using vocab mapping
        # def substitute(x):
        #     return self.vocab_mapping[x]
        
        # temp = map(substitute, input_id_array)

        # TODO: Optimize this code. Currently using for loops

        # Check if all the inputs need to be modified
        if self.data_args.vocab_modification == 'random':
            # With a 50% probability, just return the original inputs
            if np.random.uniform() < 0.5:
                return inputs


        for i in range(inputs['input_ids'].shape[0]):
            for j in range(inputs['input_ids'].shape[1]):
                inputs['input_ids'][i,j] = self.vocab_mapping[inputs['input_ids'][i,j].item()]
                if inputs['labels'][i,j] >= 0:
                    inputs['labels'][i,j] = self.vocab_mapping[inputs['labels'][i,j].item()]

        return inputs
