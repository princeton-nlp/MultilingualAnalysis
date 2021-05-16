# Changelog
All notable changes to this project will be documented in this file. Most recent change in the month occurs first.

## May 2021
1. Added `analysis/check_tokenization_split.py` for checking the expansion ration (defined in the paper).
1. Added `analysis/learn_orthogonal_mapping_one_one.py` for Procrustes method.
1. Added transliteration config files in `config/bilingual`.
1. Added `preprocessing/transliteration` for transliteration experiments.
1. Added `run_mlm_synthetic_transitive.py` for transitive syntax modification experiments.
1. Added `preprocessing/monolingual/create_relative_ratio_datasets.py`.
1. Added bilingual tokenizers (`config/bilingual`), and a file for creating them (`preprocessing/combine_tokenizers.py`).

## April 2021
1. Added file `preprocessing/combine_tokenizers.py` to combine the `vocab.json` and `merges.txt` file of two tokenizers. Duplicates are removed.
1. Added flag `--permute_words` to permute the words of a sentence randomly
1. Added script `run_experiments.py` to help running commands for different tasks.
1. `tatoeba` now uses a batch size of `1` so that the number of pad tokens is not a signal the model can exploit.
1. Added tokenizer files for different languages.
1. Added `preprocessing/qa` for converting SQuAD files to HF json.
1. Added `transformers/examples/question-answering/run_qa_synthetic.py`.
1. Downloaded FQuAD data from [here](https://fquad.illuin.tech). There is also an alternate dataset [here](https://github.com/Alikabbadj/French-SQuAD).
1. Downloaded XQuAD data from [here](https://github.com/deepmind/xquad).
1. New push April 12 (Sentence retrieval scripts).
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