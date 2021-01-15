#!/bin/bash


export DATA_DIR=../../data/RRR1994/PPAttachData.test
export TASK_NAME=ppa
export OUTPUT_DIR=ppa.output

# disable GPU
#export CUDA_VISIBLE_DEVICES=""

#--model_name_or_path bert-base-cased \

python run_ppa.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir

