# Changelog
All notable changes to this project will be documented in this file. Most recent change in the month occurs first.

## March 2021
1. Added `text-classification/run_xnli_synthetic.py` for handling synthetic modifications with XNLI. Check `scripts` folder for examples.
1. Danqi: Added `text-classification/run_glue_from_scratch.py` for handling training and evaling random model. Check `scripts/eval_random_glue.sh` for examples.

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