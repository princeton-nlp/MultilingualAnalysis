#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Different SBATCH options - https://osirim.irit.fr/site/en/articles/sbatch-options

# For salloc, use the following
# salloc --gres=gpu:1 -c 2 --mem=4G srun --pty $SHELL -l

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=xnli_hi_dep

# Remove one # to uncommment
#SBATCH --output=%x_%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=1-00:00:00    # Run for 7 days
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=asd@cs.princeton.edu

# Submit jobs
# srun python convert_dataset_to_dependency.py --language en --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/en/flattened_train_en.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/en/dep/ --task xnli &

# srun python convert_dataset_to_dependency.py --language fr --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/fr/flattened_train_fr.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/fr/dep/ --task xnli &

# srun python convert_dataset_to_dependency.py --language ar --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/ar/flattened_train_ar.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/ar/dep/ --task xnli &

# srun python convert_dataset_to_dependency.py --language hi --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/hi/flattened_train_hi.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/hi/dep/ --task xnli &

srun python convert_dataset_to_dependency.py --language en --data /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/en/flattened_train_en.json --save_dir /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/xnli/en/dep/ --task xnli &

wait;

# Finish the script
exit 0
