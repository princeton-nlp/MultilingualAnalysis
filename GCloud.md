# Instructions for running scripts with GCloud TPUs

### VM Instance
1. Use a machine which has at least 200 GB of RAM to ensure large datasets (approx 14 GB) can be run smoothly
1. Use a Deep Learning VM Image to avoid installing CUDA. There are also a few images which have Tensorflow or PyTorch installed already

### VM Setup
1. Run the `attention_guidance_setup.sh` script to set the VM up

### Storage
1. Mounting a storage bucket as a file system [link](https://cloud.google.com/storage/docs/gcs-fuse#using)
1. `gsutil` for handling google storage buckets [link](https://cloud.google.com/storage/docs/quickstart-gsutil). [Link](https://cloud.google.com/sdk/docs#linux) for installing it
1. Command for copying data - `./google-cloud-sdk/bin/gsutil cp -r ../BERT_Embeddings_Test/BERT_Embeddings_Test/word2vec_train/wiki_bookscorpus/ gs://attn-guid-europe`

### Set up TPUs
1. Tutorial for running transformers with TPUs - [link](https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch)
1. Alternate link from Google's documentation - [link](https://cloud.google.com/tpu/docs/creating-deleting-tpus#us)
1. Link for checking status and deleting TPUs - [link](https://cloud.google.com/tpu/docs/creating-deleting-tpus#ctpu_1)
