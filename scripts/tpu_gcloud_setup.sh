#!/bin/bash

function SourceCodeAndInstall {
    mkdir source_code
    cd source_code
    git clone https://github.com/ameet-1997/Multilingual.git
    cd Multilingual/transformers/
    conda activate torch-xla-1.7
    # In some instance, just `conda activate base` should work.
    # `import torch_xla` to check if it's the correct environment.
    pip install wandb
    pip install -e .
    pip install -r examples/language-modeling/requirements.txt
    pip install -r examples/token-classification/requirements.txt
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
    export VERSION=1.7
    gcloud compute tpus create tpu1 --zone=us-central1-a --network=default --version=pytorch-1.7 --accelerator-type=v3-8
    gcloud compute tpus list --zone=us-central1-a
    # export TPU_IP_ADDRESS=10.38.186.234
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
}

function RestartVM {
    conda activate torch-xla-1.7
    gcsfuse --implicit-dirs --debug_fuse multilingual-1  bucket/
    export VERSION=1.7
    gcloud compute tpus list --zone=us-central1-a
    # export TPU_IP_ADDRESS=10.38.186.234
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    cd source_code/Multilingual/transformers/
    git pull
    export WANDB_API_KEY="X"
    export WANDB_ENTITY="ameet-1997"
    export WANDB_PROJECT="multilingual_synthetic"
    export WANDB_PROJECT="multilingual_synthetic_downstream"
    export WANDB_PROJECT="multilingual_analysis"
    export WANDB_PROJECT="multilingual_mono_zs"
}

function Wandb {
    # wandb login
    export WANDB_API_KEY="X"
    export WANDB_ENTITY="ameet-1997"
    export WANDB_PROJECT="mutlilingual_word"
    # export WANDB_NAME="wikitext_mlm"
    # Run name is specified using the --run_name argument
}

function GitHub {
  # If you want your git credentials to be stored in plain text
  git config --global credential.helper store
}

function UbuntuVM {
  sudo apt install git-all
  sudo apt-get install wget
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  sudo apt install default-jre
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