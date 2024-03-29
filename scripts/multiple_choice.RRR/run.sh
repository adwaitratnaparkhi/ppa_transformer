#!/bin/bash

DATA_DIR=../../data/RRR1994/PPAttachData


set -x


for dataset in training devset test
do
  python ./extract_features.py ${DATA_DIR}/${dataset} > ./${dataset}.json
done


model=roberta-base
output_dir=results_$(date +'%s')

  python ../multiple_choice.BLBG/run_swag.py \
  --model_name_or_path ${model} \
  --train_file training.json \
  --validation_file devset.json \
  --test_file test.json \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_seq_length 80 \
  --per_gpu_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps 2 \
  --number_of_choices 2




