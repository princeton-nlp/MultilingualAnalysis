export TASK_NAME=mnli
export RUN_NO=_1

python -m pdb ../transformers/examples/text-classification/run_glue_synthetic.py \
  --model_name_or_path ../../data/model_outputs/wikitext/debug/ \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 3 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --output_dir ../../data/model_outputs/glue/debug/$TASK_NAME$RUN_NO/ \
  --run_name glue_$TASK_NAME \
  --permute_vocabulary \
  --vocab_permutation_file ../synthetic_language_files/word_based/configuration_files/permuted_vocab_seed_42_size_50265.json \
  --word_modification replace