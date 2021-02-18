#!/bin/bash

set -x

#process training data
python -m extract_features \
-i ~/ppa_transformer/data/RRR1994/PPAttachData/training \
-o ~/Downloads/tr.jsonl \
-m 1
#process test data
python -m extract_features \
-i ~/ppa_transformer/data/RRR1994/PPAttachData/devset \
-o ~/Downloads/de.jsonl \
-m 1
#process test data
python -m extract_features \
-i ~/ppa_transformer/data/RRR1994/PPAttachData/test \
-o ~/Downloads/te.jsonl \
-m 1

python -m unpooled_train \
-t ~/Downloads/tr.jsonl \
-d ~/Downloads/de.jsonl \
-e ~/Downloads/te.jsonl \
-o ~/Downloads/ \
-b