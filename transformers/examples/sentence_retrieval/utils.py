"""
Fill sentence retrieval utils here.
From Table 14 it looks like we can choose average embeddings from layer 5 of an 8 layer model.
Basically layer n/2+1, where `n` is the total number of layers.
"""

import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data.dataloader import DataLoader

def modify_config_sentence_retrieval(config):
    return config

def get_embeddings_word_modif(trainer, data_args, source_or_target):
    # We will use a batch size of 1
    # so that the number of pad tokens is not a signal for the right answer
    batch_size = 1
    if source_or_target == 'source':
        dataloader = DataLoader(
                                trainer.train_dataset,
                                # batch_size=trainer.args.train_batch_size,
                                batch_size=batch_size,
                                collate_fn=trainer.data_collator
                                )
    elif source_or_target == 'target':
        dataloader = DataLoader(
                                trainer.eval_dataset,
                                # batch_size=trainer.args.train_batch_size,
                                batch_size=batch_size,
                                collate_fn=trainer.data_collator
                                )
    else:
        raise('Wrong option in get_embeddings_word_modif')

    # Move the model to the right device
    trainer.model = trainer.model.to(trainer.args.device)
    trainer.model.eval()

    # Store the embeddings in this list
    embeddings = []
    
    for batch in dataloader:
        # Prepare the inputs
        batch = trainer._prepare_inputs(batch)
        batch['return_dict'] = True
        batch['output_hidden_states'] = True
        # Len of output hidden states is num_layers+1
        # First index corresponds to embeddings

        outputs = trainer.model(**batch)
        hidden_states = outputs.hidden_states
        # hidden_states is a tuple and each instance is of shape [batch, seq, hidden_size]

        if data_args.pool_type == 'cls':
            # Use only the representation corresponding to the first token
            hidden_states = hidden_states[-1].detach().cpu().numpy()[:,0,:]
            embeddings.append(hidden_states)
        elif data_args.pool_type == 'final':
            # Take the average of the last layer's representations
            hidden_states = hidden_states[-1].detach().cpu().numpy()
            # Take the average across dimension 1
            hidden_states = np.mean(hidden_states, axis=1)
            embeddings.append(hidden_states)
        elif data_args.pool_type == 'middle':
            # Take the average of the last layer's representations
            layer_num = (len(hidden_states) - 1)//2
            hidden_states = hidden_states[layer_num].detach().cpu().numpy()
            # Take the average across dimension 1
            hidden_states = np.mean(hidden_states, axis=1)
            embeddings.append(hidden_states)
        elif data_args.pool_type == 'higher':
            # Take the average of the last layer's representations
            layer_num = (len(hidden_states) - 1)//2 + 1
            hidden_states = hidden_states[layer_num].detach().cpu().numpy()
            # Take the average across dimension 1
            hidden_states = np.mean(hidden_states, axis=1)
            embeddings.append(hidden_states)

    # Stack all the embeddings vertically
    embeddings = np.vstack(embeddings)
    
    return embeddings

def evaluate_embeddings(source, target):
    """
    There should be a one-to-one correspondence between source and target
    """
    # Calculate cosine similarity
    cos_similarity = 1. - cdist(source, target, 'cosine')

    # Find the argmax for each row
    indices_selected = np.argmax(cos_similarity, axis=1)

    # Argmax for each index should be itself
    accuracy = np.sum(np.arange(indices_selected.shape[0]) == indices_selected) / indices_selected.shape[0] * 100

    # Return the accuracy
    return accuracy