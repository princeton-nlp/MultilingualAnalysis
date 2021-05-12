#!/bin/bash
# Author: Ameet Deshpande

# Define some global variables
SRC="en"
TGT="ar"
MODEL="../../../../bucket/model_outputs/bilingual/${SRC}_${TGT}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/bilingual/${SRC}_${TGT}/"
RUN_NAME="bi_${SRC}_${TGT}_"
ZERO="zero_"

##### XNLI #####
TASK='xnli'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$SRC/train_$SRC.json --validation_file ../../../../bucket/supervised_data/xnli/$SRC/dev_$SRC.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$SRC/train_$SRC.json --validation_file ../../../../bucket/supervised_data/xnli/$TGT/dev_$TGT.json --output_dir $OUTPUT_DIR$TASK --run_name $ZERO$RUN_NAME$TASK --model_name_or_path $OUTPUT_DIR$TASK

##### NER #####
TASK='ner'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$SRC/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$SRC/dev.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$SRC/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$TGT/dev.json --output_dir $OUTPUT_DIR$TASK --run_name $ZERO$RUN_NAME$TASK --model_name_or_path $OUTPUT_DIR$TASK

##### POS #####
TASK='pos'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$SRC/train-$SRC.json --validation_file ../../../../bucket/supervised_data/pos/$SRC/dev-$SRC.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$SRC/train-$SRC.json --validation_file ../../../../bucket/supervised_data/pos/$TGT/dev-$TGT.json --output_dir $OUTPUT_DIR$TASK --run_name $ZERO$RUN_NAME$TASK --model_name_or_path $OUTPUT_DIR$TASK

##### TATOEBA #####
TASK='tatoeba'

# Eval
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/sentence_retrieval/run_sentence_retrieval_synthetic.py --max_seq_length 128 --pool_type middle --bilingual --logging_steps 50 --overwrite_output_dir --do_train --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/tatoeba/$SRC/$SRC\_$TGT.json --output_dir $OUTPUT_DIR$TASK --run_name $ZERO$RUN_NAME$TASK --model_name_or_path $MODEL

##### XQuAD #####
TASK='xquad'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/question-answering/run_qa_synthetic.py --learning_rate 3e-5 --save_steps -1 --max_seq_length 384 --doc_stride 128 --warmup_steps 500 --weight_decay 0.0001 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../../../bucket/supervised_data/xquad/$SRC/train_$SRC.json --validation_file ../../../../bucket/supervised_data/xquad/$SRC/dev_$SRC.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/question-answering/run_qa_synthetic.py --learning_rate 3e-5 --save_steps -1 --max_seq_length 384 --doc_stride 128 --warmup_steps 500 --weight_decay 0.0001 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../../../bucket/supervised_data/xquad/$SRC/train_$SRC.json --validation_file ../../../../bucket/supervised_data/xquad/$TGT/dev_$TGT.json --output_dir $OUTPUT_DIR$TASK --run_name $ZERO$RUN_NAME$TASK --model_name_or_path $OUTPUT_DIR$TASK