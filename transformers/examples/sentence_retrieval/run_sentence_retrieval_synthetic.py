# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Running sentence retrieval for word and syntax modifications. No fine-tuning involved.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    HfArgumentParser,
    Trainer,
    TrainerWordModifications,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

# Synthetic languages
from transformers import is_wandb_available
from transformers import modify_inputs_synthetic
from transformers.synthetic_utils import modify_config
from utils import modify_config_sentence_retrieval, get_embeddings_word_modif, evaluate_embeddings

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache for training and validation data."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # Permute the vocabulary
    permute_vocabulary: bool = field(
        default=False,
        metadata={
            "help": "Whether to make the word based synthetic language which permutes the vocabulary."
        },
    )
    vocab_permutation_file: str = field(
        default=None,
        metadata={
            "help": "File which contains the mapping from the old vocabulary file to the new one. Global names are preferred"
        },
    )
    word_modification: str = field(
        default='all',
        metadata={
            "help": "all/random||add/replace"
        },
    )
    # Add, delete, or modify words
    modify_words: bool = field(
        default=False,
        metadata={
            "help": "Randomly replace words with a random word."
        },
    )
    modify_words_probability: float = field(
        default=0.15,
        metadata={
            "help": "The probability with which a word in the sentence needs to be replaced"
        },
    )
    modify_words_range: str = field(
        default='100-50000',
        metadata={
            "help": "Vocab range to sample from."
        },
    )
    # Invert the word-order
    invert_word_order: bool = field(
        default=False,
        metadata={
            "help": "Invert each sentence"
        },
    )
    # One-to-one mapping to a new vocabulary
    one_to_one_mapping: bool = field(
        default=False,
        metadata={
            "help": "Create a vocabulary with a one-to-one mapping with the new vocab, like in K. et al."
        },
    )
    one_to_one_file: str = field(
        default=None,
        metadata={
            "help": "File which contains indices in the vocabulary to ignore."
        },
    )
    shift_special: bool = field(
        default=False,
        metadata={
            "help": "When used with one-to-one mapping, also changes the [CLS] and [SEP] token. Does not change the PAD token."
        },
    )
    # Permutation
    permute_words: bool = field(
        default=False,
        metadata={
            "help": "Permute the words of the sentence randomly. Different permutation for each sentence."
        },
    )    
    # Word modification or syntax modifications
    # (If word modification, then monolingual corpus, if syntax modification, then parallel corpus)
    bilingual: bool = field(
        default=False,
        metadata={
            "help": "Is the evaluation for syntax modification or word modification. Use --bilingual for everything other than word modifications."
        },
    )
    pool_type: str = field(
        default='cls',
        metadata={
            "help": "cls/final/middle/higher.\
                    final = average of last layer \
                    middle = average of layer n/2 \
                    higher = average of layer n/2 + 1"
        },
    )    

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    if not data_args.one_to_one_mapping:
        # Don't resize the model token embeddings if we are making the one-to-one mapping modification
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        raise('This argument is hard')
    else:
        if not data_args.bilingual:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            if data_args.max_seq_length is None:
                max_seq_length = tokenizer.model_max_length
            else:
                if data_args.max_seq_length > tokenizer.model_max_length:
                    logger.warn(
                        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets_source = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            text_column_name = column_names[1]

            tokenized_datasets_target = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )            

            if data_args.max_seq_length is None:
                max_seq_length = tokenizer.model_max_length
            else:
                if data_args.max_seq_length > tokenizer.model_max_length:
                    logger.warn(
                        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)            


    # Make synthetic language modifications if necessary
    if not data_args.bilingual:
        source_datasets = deepcopy(tokenized_datasets)
        target_datasets = modify_inputs_synthetic(data_args, training_args, tokenized_datasets, tokenizer=tokenizer, task_name='tatoeba', task_type='tatoeba')
    else:
        source_datasets = tokenized_datasets_source
        target_datasets = tokenized_datasets_target

    # Initialize our Trainer
    trainer = TrainerWordModifications(
        model=model,
        args=training_args,
        data_args=data_args,
        # HACK: Using train_dataset as source and eval_dataset as target
        train_dataset=source_datasets["train"],
        eval_dataset=target_datasets["train"],
        tokenizer=tokenizer,
        # data_collator=data_collator,
    )        

    # Collect embeddings for the source and target language
    source_embeddings = get_embeddings_word_modif(trainer, data_args, 'source')
    target_embeddings = get_embeddings_word_modif(trainer, data_args, 'target')

    # Log
    logger.info('Finished extracting embeddings.')

    # Evaluate using cosine similarity as metric for KNN
    # The first index in the source dataset should be mapped to the first in target, and so on
    accuracy = evaluate_embeddings(source_embeddings, target_embeddings)
    logger.info('********')
    logger.info('The accuracy is: {}%'.format(accuracy))
    logger.info('********')

    # Log the accuracy using wandb
    if is_wandb_available():
        import wandb
        # Initialize wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            config=vars(data_args),
            name=training_args.run_name,
        )        
        log_dict = {'accuracy': accuracy}
        wandb.log(log_dict)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
