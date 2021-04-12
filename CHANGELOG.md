# Changelog
All notable changes to this project will be documented in this file. Most recent change in the month occurs first.

## April 2021
1. Added files to `transformers/examples/sentence_retrieval`. Commands in `Debug.md`.
1. New push April 10 (Sentence retrieval and one to one mapping). [Link](https://github.com/ameet-1997/Multilingual/commit/4ecdb1e66981ecb6390afb344a72ed1995978843)
1. Added a  file to create indices to ignore in one_to_one mapping: `synthetic_language_files/word_based/one_to_one_mapping.py`.
1. Added preprocessing scripts for sentence retrieval. `preprocessing/sentence_retrieval`.
1. New push on April 9 (NER and POS preprocessing). [Link](https://github.com/ameet-1997/Multilingual/commit/9456acdb4cc4ef46d634a883e5ba94cf1fb1cded).
1. Added preprocessing scripts to `preprocessing/ner_pos`.
1. New push on April 8 (XNLI preprocessing). [Link](https://github.com/ameet-1997/Multilingual/commit/649d916b99e0cbcda3fa8e14c390fcc15af954a4).
1. `supervised_dataset_loop.slurm` in `preprocessing/gdtreebank/` for looping over languages and submitting jobs.
1. Directory `preprocessing/ner` changed to `preprocessing/ner_pos`
1. XNLI data stored in `data/xnli`.
1. Added `preprocesssing/xnli`. `convert_xnli_tsv_to_json.py` and other files added. 
1. Debugged word modifications for POS. Commands in `Debug.md`.
1. Modified `synthetic_utils.py` to work with NER and POS.

## March 2021
1. Added `preprocessing/ner` for pre-processing POS and NER datasets to JSON.
1. Added an `experiments` folder to keep track of all the experiments being performed.
1. Added `convert_dataset_to_sentences.py` in `preprocessing/dependency_parsing/`.
1. Added `convert_conllu_to_corpus.py` in `preprocessing/dependency_parsing/`.
1. Added `scripts/one_to_one` for GLUE and XNLI scripts for one-to-one mapping. Best to create folders for each kind of modification.
1. Added `preprocessing/gdtreebank/` for galactic dependencies.
1. Added `preprocessing/dependency_parsing/` for parsing corpus using `stanza`.
1. Added `text-classification/run_xnli_synthetic.py` for handling synthetic modifications with XNLI. Check `scripts` folder for examples.

## February 2021
1. Modified `modify_inputs_invert` function to work both with MLM and MNLI.
1. Added function `modify_inputs_invert` in `synthetic_utils.py` to invert the sentence (Dufter et al.).

## January 2021
1. Added file called `synthetic_utils.py` in `examples/language-modeling` as an alternative to `synthetic_language_modifications_utils.py`.
1. To make word based synthetic modifications, a file called `run_mlm_synthetic.py` has been added to `examples/language-modeling`. The `train` function in `TrainerWordModifications` instantiates `WordBasedModifications` from `synthetic_language_modifications_utils.py` and makes the modifications needed to the `inputs` being passed for processing.
1. Added file `synthetic_language_modifications_utils.py` in folder `transformers/src/transformers` to define classes and functions useful for the modifications.
1. Added directory `synthetic_language_files` to store configuration files and scripts for creating synthetic language transformations.
1. Added `trainer_word_modifications.py` in `transformers/src/transformers` for word based modifications to the datasets.
1. Added `run_glue_tpu.sh` script in `scripts` folder.