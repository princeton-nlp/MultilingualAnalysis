export TASK_NAME=xnli_fr
export RUN_NAME=xnli_fr_eval
export RUN_NO=_invert

python ../transformers/examples/xla_spawn.py --num_cores 1 ../transformers/examples/text-classification/run_xnli_synthetic.py \
  --model_name_or_path ../../../bucket/model_outputs/xnli/$TASK_NAME$RUN_NO/ \
  --language fr \
  --cache_dir=../../../bucket/cache \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --output_dir ../../../bucket/model_outputs/xnli/eval_$TASK_NAME$RUN_NO/ \
  --run_name $RUN_NAME \
  --invert_word_order \
  --word_modification add  