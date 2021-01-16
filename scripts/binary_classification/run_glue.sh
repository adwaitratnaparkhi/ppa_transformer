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

