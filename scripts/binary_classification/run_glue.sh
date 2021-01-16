#!/bin/bash

export DATA_DIR=../../data/RRR1994/PPAttachData.test

function mk_data
{
  for x in training devset test
  do
    python ../data_prep/heads_to_json.py < ${DATA_DIR}/${x} > ${x}.json
  done
}

mk_data

export CUDA_VISIBLE_DEVICES=""

python run_glue.py \
  --model_name_or_path roberta-base \
  --train_file training.json \
  --validation_file devset.json \
  --test_file test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./ppa.out

  #--no_use_fast_tokenizer
