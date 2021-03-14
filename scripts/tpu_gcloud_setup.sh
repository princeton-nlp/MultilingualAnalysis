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
    gcsfuse --implicit-dirs --debug_fuse multilingual-1  bucket/
    export VERSION=1.6
    gcloud compute tpus list --zone=europe-west4-a
    # export TPU_IP_ADDRESS=10.38.186.234
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    cd source_code/Multilingual/transformers/
    export WANDB_API_KEY="fc68097ba21d58900b072a1279cf3cf6d83eb0c6"
    export WANDB_ENTITY="ameet-1997"
    export WANDB_PROJECT="mutlilingual_word"
}

function Danqi_RestartVM {
    conda activate torch-xla-1.6
    gcsfuse --implicit-dirs --debug_fuse multilingual-2  bucket/
    export VERSION=1.6
    gcloud compute tpus create tpu3 --zone=us-central1-f --network=default --version=pytorch-1.6 --accelerator-type=v2-8
    gcloud compute tpus list --zone=us-central1-f
    # export TPU_IP_ADDRESS=10.38.186.234
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    cd source_code/Multilingual/transformers/
    export WANDB_API_KEY="c506f6f94ffd4e28768ff0cf1637bd2a28902a32"
    export WANDB_ENTITY="danqi7"
    export WANDB_PROJECT="mutlilingual_random"
}

function Wandb {
    # wandb login
    export WANDB_API_KEY="fc68097ba21d58900b072a1279cf3cf6d83eb0c6"
    export WANDB_ENTITY="ameet-1997"
    export WANDB_PROJECT="mutlilingual_word"
    # export WANDB_NAME="wikitext_mlm"
    # Run name is specified using the --run_name argument
}

###

for arg in "$@"; do
  if [[ "$arg" = -i ]] || [[ "$arg" = --initial ]]; then
    ARG_INITIAL=true
    ARG_RESTART=false
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