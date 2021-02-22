#!/bin/bash

set -x
DATA_DIR=~/pp-attachment/dataset/Belinkov2014/pp-data-english/

#process training data
python -m extract_features \
-i ${DATA_DIR}/wsj.2-21.txt.dep.pp \
-o tr.jsonl \
-m 2

#process test data
python -m extract_features \
-i ${DATA_DIR}/wsj.23.txt.dep.pp \
-o te.jsonl \
-m 2

#run training
python -m unpooled_train \
-t tr.jsonl \
-d te.jsonl \
-o ./