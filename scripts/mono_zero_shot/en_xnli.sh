LANG="en"
TARGET="ar"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_to_${TARGET}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

TASK='xnli'

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dep/synthetic_dep_flattened_dev_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD

LANG="en"
TARGET="fr"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_to_${TARGET}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dep/synthetic_dep_flattened_dev_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


LANG="en"
TARGET="hi"
EVAL="_zero"
MODEL_DIR="monolingual_500k"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_to_${TARGET}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dep/synthetic_dep_flattened_dev_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD