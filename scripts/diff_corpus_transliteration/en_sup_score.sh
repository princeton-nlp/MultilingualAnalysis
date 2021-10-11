#!/bin/bash
# Author: Ameet Deshpande

# Define some global variables
LANG="en"
EVAL="_zero"
MODEL_DIR="one_to_one_diff_source_100"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/sup/"
RUN_NAME="sup_${LANG}_${MODEL_DIR}_"
ZERO_SHOT_ADD="--one_to_one_mapping --word_modification replace"

##### XNLI #####
TASK='xnli'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dev_$LANG.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL $ZERO_SHOT_ADD


##### NER #####
TASK='ner'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dev.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL $ZERO_SHOT_ADD

##### POS #####
TASK='pos'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dev-$LANG.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL $ZERO_SHOT_ADD

##### XQuAD #####
TASK='xquad'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/question-answering/run_qa_synthetic.py --learning_rate 3e-5 --save_steps -1 --max_seq_length 384 --doc_stride 128 --warmup_steps 500 --weight_decay 0.0001 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../../../bucket/supervised_data/xquad/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xquad/$LANG/dev_$LANG.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL $ZERO_SHOT_ADD