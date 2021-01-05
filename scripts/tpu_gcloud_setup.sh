#!/bin/bash

function SourceCodeAndInstall {
    mkdir source_code
    cd source_code
    git clone https://github.com/ameet-1997/Multilingual.git
    cd Multilingual/transformers/
    conda activate torch-xla-1.6
    pip install wandb
    pip install -e .
    pip install -r examples/language-modeling/requirements.txt
}

function GCFuse {
    # Install gcsfuse
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    printf 'Y' | sudo apt-get install gcsfuse
    sudo apt-get install htop
}

function MountBucket {
    # Mount the bucket for saving model files
    cd ~
    mkdir bucket
    gcsfuse --implicit-dirs multilingual-1  bucket/
    cd source_code/Multilingual/transformers/
}

function MakeTPUs {
    export VERSION=1.6
    gcloud compute tpus create tpu1 --zone=europe-west4-a --network=default --version=pytorch-1.6 --accelerator-type=v3-8
    gcloud compute tpus list --zone=europe-west4-a
    # export TPU_IP_ADDRESS=10.38.186.234
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
}

function RestartVM {
    conda activate torch-xla-1.6
    gcsfuse --implicit-dirs multilingual-1  bucket/
    export VERSION=1.6
    gcloud compute tpus list --zone=europe-west4-a
    # export TPU_IP_ADDRESS=10.38.186.234
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    cd source_code/Multilingual/transformers/
}

function Wandb {
    # wandb login
    export WANDB_API_KEY="fc68097ba21d58900b072a1279cf3cf6d83eb0c6"
    export WANDB_ENTITY="ameet-1997"
    export WANDB_PROJECT="mutlilingual_word"
    export WANDB_NAME="wikitext_mlm"
}

###

for arg in "$@"; do
  if [[ "$arg" = -i ]] || [[ "$arg" = --initial ]]; then
    ARG_INITIAL=true
  fi
  if [[ "$arg" = -r ]] || [[ "$arg" = --restart ]]; then
    ARG_RESTART=true
    ARG_INITIAL=false
  fi
done

###

if [[ "$ARG_INITIAL" = true ]]; then
  SourceCodeAndInstall
  GCFuse
  MountBucket
  Wandb
fi

if [[ "$ARG_RESTART" = true ]]; then
  RestartVM
  Wandb
fi

# # Install Conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

# # # Install the required conda environment
# # conda env create -f BERT_Embeddings_Test/yaml_files/ag.yml
# cd BERT_Embeddings_Test/experiments/attention_guidance/transformers/
# conda activate torch-xla-1.5
# pip install wandb
# pip install -e .

# # Copy required dataset from the bucket
# cd ~
# mkdir data
# cd data
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bert_full.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bert_val.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bert_train.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/wikicorpus_en_one_article_per_line.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bookscorpus_one_book_per_line.txt .

# # Install gcsfuse
# export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
# echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install gcsfuse

# # Install htop
# sudo apt-get install htop

# # Mount the bucket for saving model files
# cd ~
# mkdir bucket
# gcsfuse --implicit-dirs atto-guid-europe  bucket/

# # Go to the source code home dir
# cd source_code/BERT_Embeddings_Test/experiments/attention_guidance/transformers/

# # Setting up a TPU
# export VERSION=1.5
# gcloud compute tpus create tpu1 --zone=europe-west4-a --network=default --version=pytorch-1.5 --accelerator-type=v3-8
# gcloud compute tpus list --zone=europe-west4-a
# # export TPU_IP_ADDRESS=10.38.186.234
# # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# # Log in to wandb
# wandb login


# # When restarting instance
# conda activate torch-xla-1.5
# gcsfuse --implicit-dirs atto-guid-europe  bucket/
# export VERSION=1.5
# gcloud compute tpus list --zone=europe-west4-a
# # export TPU_IP_ADDRESS=10.38.186.234
# # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
# cd source_code/BERT_Embeddings_Test/experiments/attention_guidance/transformers/
# wandb login