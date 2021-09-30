# Instructions for running scripts with GCloud TPUs

### VM Instance
1. Use a machine which has at least 200 GB of RAM to ensure large datasets (approx 14 GB) can be run smoothly
1. Use a Deep Learning VM Image to avoid installing CUDA. There are also a few images which have Tensorflow or PyTorch installed already

### VM Setup
1. Run the `scripts/tpu_gcloud_setup.sh` script to set the VM up - `tpu_gcloud_setup.sh -i` or `tpu_gcloud_setup.sh -r`

### Storage
1. Mounting a storage bucket as a file system [link](https://cloud.google.com/storage/docs/gcs-fuse#using)
1. `gsutil` for handling google storage buckets [link](https://cloud.google.com/storage/docs/quickstart-gsutil). [Link](https://cloud.google.com/sdk/docs#linux) for installing it
1. Command for copying data - `./google-cloud-sdk/bin/gsutil cp /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/dependency_parse_data/english/mono_dep_en_train.txt  gs://multilingual-1/syntax_modified_data/english`. `google-cloud-sdk` is in `/n/fs/nlp-asd/asd/asd/cloud`.

### Increasing disk size
1. [Link](https://www.cloudbooklet.com/how-to-resize-disk-of-a-vm-instance-in-google-cloud/) for increase/resizing disk.
1. If you are resizing a boot disk, you might have to restart the VM. If you are using a non-boot disk, then you might have to perform some additional operations. [(link)](https://cloud.google.com/compute/docs/disks/working-with-persistent-disks?authuser=1#resize_pd)
1. [Link](https://cloud.google.com/sdk/gcloud/reference/compute/disks/resize?hl=en) for programmatically doing it. I haven't tried this method.

### Set up TPUs
1. Tutorial for running transformers with TPUs - [link](https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch)
1. Alternate link from Google's documentation - [link](https://cloud.google.com/tpu/docs/creating-deleting-tpus#us)
1. Link for checking status and deleting TPUs - [link](https://cloud.google.com/tpu/docs/creating-deleting-tpus#ctpu_1)

### SSH details
1. Your identification has been saved in /n/fs/grad/asd/.ssh/google_compute_engine.
1. Your public key has been saved in /n/fs/grad/asd/.ssh/google_compute_engine.pub.

### Using GCloud SDK with your terminal
1. Change to cloud directory - `cd /n/fs/nlp-asd/asd/asd/cloud`
1. Initialize configuration along with default project, region, and zone - `./google-cloud-sdk/bin/gcloud init`
1. Connect to a VM - `./google-cloud-sdk/bin/gcloud compute ssh --project=attention-guidance --zone=europe-west4-a mult-p-1`
1. Start a VM -`./google-cloud-sdk/bin/gcloud compute instances start --project=attention-guidance --zone=europe-west4-a  mult-p-1`
1. Stop a VM - `./google-cloud-sdk/bin/gcloud compute instances stop --project=attention-guidance --zone=europe-west4-a  mult-p-1`
1. List all Google Cloud instances - `./google-cloud-sdk/bin/gcloud compute instances list`

### Commands for zone us-central1-a
1. Connect to a VM - `./google-cloud-sdk/bin/gcloud compute ssh --project=attention-guidance --zone=us-central1-a exp-1`
1. Start a VM -`./google-cloud-sdk/bin/gcloud compute instances start --project=attention-guidance --zone=us-central1-a  exp-1`
1. Stop a VM - `./google-cloud-sdk/bin/gcloud compute instances stop --project=attention-guidance --zone=us-central1-a  exp-1`