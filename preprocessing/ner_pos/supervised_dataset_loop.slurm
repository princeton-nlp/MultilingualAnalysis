#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Different SBATCH options - https://osirim.irit.fr/site/en/articles/sbatch-options

# For salloc, use the following
# salloc --gres=gpu:1 -c 2 --mem=4G srun --pty $SHELL -l

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=conllu_ner_pos

# Remove one # to uncommment
#SBATCH --output=%x_%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=0-01:00:00
# #SBATCH --nodelist=node913
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=asd@cs.princeton.edu

# Submit jobs
language_string="en fr ar hi"
for language1 in $language_string; do
for language2 in $language_string; do
# You can use srun to run multiple scripts in the same job in parallel (make sure to use & at the end!). Note how you can specify the resources used for each srun and make them exclusive using the --exclusive flag.
    # srun python convert_dataset_to_dependency.py --language ${language1} --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/ner/${language1}/flattened_dev.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/ner/${language1}/dep/ --task ner
    # srun python convert_dataset_to_dependency.py --language ${language1} --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/pos/${language1}/flattened_dev-${language1}.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/pos/${language1}/dep/ --task pos
    srun python convert_galactic_to_corpus.py --galactic_file /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/ner/${language1}/dep/dep_flattened_dev-${language1}~${language2}@N~${language2}@V.conllu --supervised_dataset /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/ner/${language1}/dev.json --index_selector /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/ner/${language1}/dep/selected_indices_flattened_dev.json --task ner
    srun python convert_galactic_to_corpus.py --galactic_file /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/pos/${language1}/dep/dep_flattened_dev-${language1}-${language1}~${language2}@N~${language2}@V.conllu --supervised_dataset /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/pos/${language1}/dev-${language1}.json --index_selector /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/pos/${language1}/dep/selected_indices_flattened_dev-${language1}.json --task pos
done;
done;

wait;

# Finish the script
exit 0
