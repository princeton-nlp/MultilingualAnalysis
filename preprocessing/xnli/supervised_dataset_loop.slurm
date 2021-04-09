#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Different SBATCH options - https://osirim.irit.fr/site/en/articles/sbatch-options

# For salloc, use the following
# salloc --gres=gpu:1 -c 2 --mem=4G srun --pty $SHELL -l

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=conllu_to_xnli

# Remove one # to uncommment
#SBATCH --output=%x_%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
# #SBATCH --nodelist=node913
# #SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=asd@cs.princeton.edu

# Submit jobs
language_string="en fr ar hi"
for language1 in $language_string; do
for language2 in $language_string; do
# You can use srun to run multiple scripts in the same job in parallel (make sure to use & at the end!). Note how you can specify the resources used for each srun and make them exclusive using the --exclusive flag.
    srun python convert_galactic_to_corpus.py --galactic_file /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/${language1}/dep/dep_flattened_dev_${language1}-${language1}~${language2}@N~${language2}@V.conllu --supervised_dataset /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/${language1}/dev_${language1}.json --index_selector /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/${language1}/dep/selected_indices_flattened_dev_${language1}.json --task xnli
done;
done;

wait;

# Finish the script
exit 0
