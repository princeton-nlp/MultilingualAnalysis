export TASK_NAME=qnli
export RUN_NO=_random
export OUTPUT_DIR=../../../bucket/model_outputs/glue/$TASK_NAME$RUN_NO/

# Random: --from_scratch + --config_name + --tokenizer_name
python ../transformers/examples/xla_spawn.py --num_cores 1 ../transformers/examples/text-classification/run_glue_from_scratch.py \
  --from_scratch \
  --config_name roberta-base \
  --tokenizer_name roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir  $OUTPUT_DIR \
  --overwrite_output_dir \
  --run_name glue_$TASK_NAME$RUN_NO


# Pre-trained: --model_name_or_path
# python ../transformers/examples/text-classification/run_glue.py \
#   --model_name_or_path roberta-base \
#   --task_name $TASK_NAME \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir  $OUTPUT_DIR \
#   --overwrite_output_dir \
#   --run_name glue_$TASK_NAME$RUN_NO