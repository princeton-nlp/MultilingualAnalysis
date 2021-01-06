export TASK_NAME=mnli
export RUN_NO=_1

python ../transformers/examples/xla_spawn.py --num_cores 1 ../transformers/examples/run_glue.py \
  --model_name_or_path ../../../bucket/model_outputs/wikitext/mono_english/ \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ../../../bucket/model_outputs/glue/$TASK_NAME$RUN_NO/