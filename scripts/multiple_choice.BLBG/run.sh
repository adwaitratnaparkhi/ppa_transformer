#!/bin/bash

DATA_DIR=../../data/BLBG2014/pp-data-english


set -x

export PYTHONPATH=../data_prep

for experiment_mode in 1 2 3
 do
  python ./extract_features.py ${DATA_DIR}/wsj.2-21.txt.dep.pp ${experiment_mode} > tr.${experiment_mode}.json
  python ./extract_features.py ${DATA_DIR}/wsj.23.txt.dep.pp ${experiment_mode}   > te.${experiment_mode}.json

  output_dir=results_$(date +'%s')

  python ./run_swag.py \
  --model_name_or_path roberta-base \
  --train_file tr.${experiment_mode}.json \
  --test_file te.${experiment_mode}.json \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --do_train \
  --do_predict \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_seq_length 80 \
  --per_gpu_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps 2 \
  --number_of_choices 8

  done


