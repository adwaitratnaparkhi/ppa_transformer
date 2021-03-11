#!/bin/bash

set -x
DATA_DIR=../../data/RRR1994/PPAttachData

#process training data
python -m extract_features \
-i ${DATA_DIR}/training \
-o tr.jsonl \
-m 1

#process test data
python -m extract_features \
-i ${DATA_DIR}/devset \
-o de.jsonl \
-m 1

#process test data
python -m extract_features \
-i ${DATA_DIR}/test \
-o te.jsonl \
-m 1

python -m unpooled_train \
-t tr.jsonl \
-d de.jsonl \
-e te.jsonl \
-o ./ \
-m 1