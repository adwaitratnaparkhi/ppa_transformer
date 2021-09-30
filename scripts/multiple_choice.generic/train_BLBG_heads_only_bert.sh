#!/bin/bash

set -x

DATA_DIR=../../data/BLBG2014/pp-data-english

#process training data
python -m extract_features \
-i ${DATA_DIR}/wsj.2-21.txt.dep.pp \
-o tr.jsonl \
-m 3

#process test data
python -m extract_features \
-i ${DATA_DIR}/wsj.23.txt.dep.pp \
-o te.jsonl \
-m 3

#run training
python -m unpooled_train \
-t tr.jsonl \
-d te.jsonl \
-e te.jsonl \
-o ./ \
-m 2