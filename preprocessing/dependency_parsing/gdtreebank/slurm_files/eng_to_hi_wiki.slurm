#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Different SBATCH options - https://osirim.irit.fr/site/en/articles/sbatch-options

# For salloc, use the following
# salloc --gres=gpu:1 -c 2 --mem=4G srun --pty $SHELL -l

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=galactic_en_to_hi

# Remove one # to uncommment
#SBATCH --output=%x_%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=10
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=node913
# #SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=asd@cs.princeton.edu

# Submit jobs
srun bin/gd-translate --input /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/dependency_parse_data/english/dep_wiki.train.raw --spec en~hi@N~hi@V --mem 230g &
# srun bin/gd-translate --input ../../../../data/dependency_parse_data/english/dep_wiki_train.txt --spec en~hi@N~hi@V --mem 60g &

wait;

# Finish the script
exit 0
