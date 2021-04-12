export LANGUAGE=hindi
export MODEL_COFIG=../../config/roberta_8/roberta_base_8_512.json

python examples/xla_spawn.py -num_cores 8 examples/language-modeling/run_mlm_synthetic.py \
    --train_file ../../../../bucket/hindi_wiki/wiki.train.txt \
    --validation_file ../../../../bucket/hindi_wiki/wiki.valid.txt \
    --cache_dir ../../../../bucket/cache \
    --output_dir ../../../../bucket/model_outputs/hindi/hindi_word_modification_inverted \ 
    --model_type roberta \
    --config_name $MODEL_COFIG \
    --tokenizer_name ../../config/hindi \
    --learning_rate 1e-4 \
    --num_train_epochs 40 \
    --warmup_steps 10000 \
    --do_train \
    --do_eval \
    --save_steps 10000 \
    --logging_steps 50 \
    --per_device_train_batch_size 16 \ 
    --overwrite_output_dir \
    --run_name inverted_sentence_240_$LANGUAGE \
    --invert_word_order \
    --word_modification add
