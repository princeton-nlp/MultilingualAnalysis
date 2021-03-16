export TASK_NAME=mnli
export RUN_NO=_1

python -m pdb ../../transformers/examples/text-classification/run_glue_synthetic.py \
  --model_name_or_path ../../../data/model_outputs/wikitext/debug/ \
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
  --cache_dir /n/fs/nlp-asd/asd/asd/BERT_Embeddings_Test/BERT_Embeddings_Test/global_data/transformer_models \
  --output_dir ../../../data/model_outputs/glue/debug/$TASK_NAME$RUN_NO/ \
  --run_name glue_$TASK_NAME \
  --one_to_one_mapping \
  --word_modification replace
