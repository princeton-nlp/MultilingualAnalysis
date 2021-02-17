export TASK_NAME=mnli
export RUN_NO=_random

python ../transformers/examples/xla_spawn.py --num_cores 1 ../transformers/examples/text-classification/run_glue.py \
  --model_name_or_path ~/bucket/model_outputs/wikitext/random/\
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --seed 42 \
  --output_dir ../../../bucket/model_outputs/glue/$TASK_NAME$RUN_NO/ \
  --run_name glue_$TASK_NAME$RUN_NO