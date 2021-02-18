#!/bin/bash

set -x

#process training data
python -m extract_features \
-i ~/pp-attachment/dataset/Belinkov2014/pp-data-english/wsj.2-21.txt.dep.pp \
-o ~/Downloads/tr.jsonl \
-m 2
#process test data
python -m extract_features \
-i ~/pp-attachment/dataset/Belinkov2014/pp-data-english/wsj.23.txt.dep.pp \
-o ~/Downloads/te.jsonl \
-m 2
#run training
python -m unpooled_train \
-t ~/Downloads/tr.jsonl \
-d ~/Downloads/te.jsonl \
-o ~/Downloads/