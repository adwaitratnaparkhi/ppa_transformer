#!/bin/bash


DATA_DIR=../../data/BLBG2014/pp-data-english

experiment_mode=3

set -x

python ./extract_features.py ${DATA_DIR}/wsj.2-21.txt.dep.pp ${experiment_mode} | head -10 > tr.json
python ./extract_features.py ${DATA_DIR}/wsj.23.txt.dep.pp ${experiment_mode} | head -10 > te.json

python ./run_swag.py \
--model_name_or_path roberta-base \
--train_file tr.json \
--validation_file te.json \
--output_dir ./ppa.out \
--overwrite_output_dir \
--do_train \
--do_eval
