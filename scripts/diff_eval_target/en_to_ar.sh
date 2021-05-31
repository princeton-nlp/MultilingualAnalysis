#!/bin/bash
# Author: Ameet Deshpande

# Define some global variables
LANG="en"
TARGET="ar"
EVAL="_zero"
MODEL_DIR="syntax_modif_${TARGET}"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""
DIFF_EVAL="_diff"

##### XNLI #####
TASK='xnli'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/dep/synthetic_dep_flattened_train_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dep/synthetic_dep_flattened_dev_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$DIFF_EVAL --run_name $RUN_NAME$TASK$DIFF_EVAL --model_name_or_path $MODEL


##### NER #####
TASK='ner'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/dep/synthetic_dep_flattened_train-$LANG~$TARGET\@N~$TARGET\@V.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dep/synthetic_dep_flattened_dev-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$DIFF_EVAL --run_name $RUN_NAME$TASK$DIFF_EVAL --model_name_or_path $MODEL

##### POS #####
TASK='pos'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/dep/synthetic_dep_flattened_train-$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dep/synthetic_dep_flattened_dev-$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$DIFF_EVAL --run_name $RUN_NAME$TASK$DIFF_EVAL --model_name_or_path $MODEL