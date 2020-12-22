mkdir source_code
cd source_code
git clone https://github.com/ameet-1997/BERT_Embeddings_Test.git

# # Install Conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

# # Install the required conda environment
# conda env create -f BERT_Embeddings_Test/yaml_files/ag.yml
cd BERT_Embeddings_Test/experiments/attention_guidance/transformers/
conda activate torch-xla-1.5
pip install wandb
pip install -e .

# # Copy required dataset from the bucket
# cd ~
# mkdir data
# cd data
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bert_full.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bert_val.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bert_train.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/wikicorpus_en_one_article_per_line.txt .
# gsutil cp gs://atto-guid-europe/wiki_bookscorpus/bookscorpus_one_book_per_line.txt .

# Install gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

# Install htop
sudo apt-get install htop

# Mount the bucket for saving model files
cd ~
mkdir bucket
gcsfuse --implicit-dirs atto-guid-europe  bucket/

# Go to the source code home dir
cd source_code/BERT_Embeddings_Test/experiments/attention_guidance/transformers/

# Setting up a TPU
export VERSION=1.5
gcloud compute tpus create tpu1 --zone=europe-west4-a --network=default --version=pytorch-1.5 --accelerator-type=v3-8
gcloud compute tpus list --zone=europe-west4-a
# export TPU_IP_ADDRESS=10.38.186.234
# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# Log in to wandb
wandb login


# When restarting instance
conda activate torch-xla-1.5
gcsfuse --implicit-dirs atto-guid-europe  bucket/
export VERSION=1.5
gcloud compute tpus list --zone=europe-west4-a
# export TPU_IP_ADDRESS=10.38.186.234
# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
cd source_code/BERT_Embeddings_Test/experiments/attention_guidance/transformers/
wandb login