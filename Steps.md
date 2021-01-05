# Steps for differents parts of the pipeline

### Preprocessing data
1. Scripts can be found in `preprocessing` directory
1. The cached data gets stored in - `/home/asd/.cache/huggingface/datasets/text/`. So make sure there is enough space on the disk

### Running code on TPUs
1. MLM Monolingual - `python examples/xla_spawn.py --num_cores 8 examples/language-modeling/run_mlm.py --output_dir=../../../bucket/model_outputs/wikitext/mono_english --model_type=roberta --config_name=../config/roberta_8_orig/ --tokenizer_name=../config/roberta_8_orig/ --num_train_epochs 20 --do_train --do_eval  --train_file=../../../bucket/wikitext-103-raw/wiki.train.txt --validation_file=../../../bucket/wikitext-103-raw/wiki.valid.txt --save_steps 10000`
1. MLM RoBERTa-base - `nohup   python examples/xla_spawn.py --num_cores 8 examples/language-modeling/run_mlm.py --train_file=../../../bucket/wikitext-103-raw/wiki.train.txt --validation_file=../../../bucket/wikitext-103-raw/wiki.valid.txt --output_dir=../../../bucket/model_outputs/wikitext/mono_english --model_type=roberta --config_name=roberta-base --tokenizer_name=roberta-base --learning_rate 1e-4 --num_train_epochs 20 --do_train --do_eval --save_steps 10000 --logging_steps 50 --per_device_train_batch_size 16 --overwrite_output_dir   &`