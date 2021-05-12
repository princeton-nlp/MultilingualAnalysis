LANG="hi"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_inverted_"
ZERO_SHOT_ADD="--invert_word_order --word_modification replace"

TASK='pos'

# Inverted
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dev-$LANG.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_permute_"
ZERO_SHOT_ADD="--permute_words --word_modification replace"

# Permute
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dev-$LANG.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_word_modif_"
ZERO_SHOT_ADD="--modify_words --modify_words_probability 0.15 --word_modification replace"

# Word modif 15
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dev-$LANG.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_one_"
ZERO_SHOT_ADD="--one_to_one_mapping --word_modification replace"

# One to one mapping
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dev-$LANG.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
TARGET="ar"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_to_${TARGET}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

# hi_to_ar
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dep/synthetic_dep_flattened_dev-$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
TARGET="en"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_to_${TARGET}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

# hi_to_en
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dep/synthetic_dep_flattened_dev-$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="hi"
TARGET="fr"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_to_${TARGET}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

# hi_to_fr
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dep/synthetic_dep_flattened_dev-$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD