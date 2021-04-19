#!/bin/bash



# Avoid taking up too much disk space
function cleanup {
    rm results_*/pytorch_model.bin
    rm results_*/checkpoint-*/pytorch_model.bin
    rm results_*/checkpoint-*/optimizer.pt
}

set -x

cd multiple_choice.RRR
./run.sh
cleanup

cd ../multiple_choice.BLBG
./run.sh
cleanup

cd ../multiple_choice.generic

./train_BLBG_full_sentence_bert.sh
cleanup

./train_BLBG_full_sentence_roberta.sh
cleanup

./train_BLBG_heads_only_bert.sh
cleanup

./train_BLBG_heads_only_roberta.sh
cleanup

./train_RRR_bert.sh
cleanup

./train_RRR_roberta.sh
cleanup





