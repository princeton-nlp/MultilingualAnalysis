# Define some global variables
LANG="en"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_inverted_"
ZERO_SHOT_ADD="--invert_word_order --word_modification replace"
TASK='ner'


python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic_corrected_labels.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dev.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="ar"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_inverted_"
ZERO_SHOT_ADD="--invert_word_order --word_modification replace"

# Zero-shot eval
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic_corrected_labels.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dev.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_inverted_"
ZERO_SHOT_ADD="--invert_word_order --word_modification replace"


python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic_corrected_labels.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dev.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD



LANG="fr"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_inverted_"
ZERO_SHOT_ADD="--invert_word_order --word_modification replace"

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic_corrected_labels.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dev.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD
