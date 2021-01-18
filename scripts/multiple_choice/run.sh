#!/bin/bash

DATA_DIR=../../data/BLBG2014/pp-data-english

experiment_mode=1

set -x

export PYTHONPATH=../data_prep

python ./extract_features.py ${DATA_DIR}/wsj.2-21.txt.dep.pp ${experiment_mode} > tr.json
python ./extract_features.py ${DATA_DIR}/wsj.23.txt.dep.pp ${experiment_mode}  > te.json

# to disable GPU
#export CUDA_VISIBLE_DEVICES=""

python ./run_swag.py \
--model_name_or_path roberta-base \
--train_file tr.json \
--validation_file te.json \
--output_dir ./ppa.out \
--overwrite_output_dir \
--do_train \
--do_eval \
--save_total_limit 1 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2

