export TASK_NAME=mnli
export RUN_NO=_1

# Note that no task_name is passed

python -m pdb ../../transformers/examples/text-classification/run_glue_synthetic.py \
  --model_name_or_path ../../../data/model_outputs/wikitext/debug/ \
  --train_file /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/dependency_parse_data/english/MNLI/en_hi_N_hi_V/mono_dep_flattened_dev_matched-en~hi@N~hi@V.json \
  --validation_file /n/fs/nlp-asd/asd/asd/Projects/Multilingual/data/dependency_parse_data/english/MNLI/en_hi_N_hi_V/mono_dep_flattened_dev_mismatched-en~hi@N~hi@V.json \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 3 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --cache_dir /n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models \
  --output_dir ../../../data/model_outputs/glue/debug/$TASK_NAME$RUN_NO/ \
  --run_name glue_$TASK_NAME \
  --invert_word_order \
  --word_modification replace
