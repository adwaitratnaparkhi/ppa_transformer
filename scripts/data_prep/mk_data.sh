#!/bin/bash

set -x

# This is the Penn Treebank Release 3 installation directory
PENN_TREEBANK_DIR="../../../../data/treebank_3"

# This is the installation dir of the code and scripts from (Belinkov, 2014), available at https://github.com/boknilev/pp-attachment
BLBG_DIR="../../../pp-attachment"


WSJDIR="${PENN_TREEBANK_DIR}/parsed/mrg/wsj"

DATA_DIR="../../data/Belinkov2014/pp-data-english"
DATA_DIR_REBUILT="${DATA_DIR}.rebuilt"

TR_DEP="tb3.2-21.dep"
TE_DEP="tb3.23.dep"

function mk_local_copy {

    mkdir -p ${DATA_DIR}
    
    cp  ${BLBG_DIR}/data/pp-data-english/* ${DATA_DIR}
}



# Rebuild the Belinkov dataset
function mkdata1 {
    ./tb3_to_dependency.sh ${WSJDIR} ${TR_DEP} ${TE_DEP}

    mkdir -p ${DATA_DIR_REBUILT}

    python2 ./extract_pp_attach_for_matlab.py ${TR_DEP} ${TE_DEP} ${DATA_DIR_REBUILT}/wsj.2-21.txt.dep.pp  ${DATA_DIR_REBUILT}/wsj.23.txt.dep.pp
}



# Join sentences to original based on rebuilt dataset
function mkdata2 {

    for sections in 2-21 23
    do
	python ./join_sentence_from_rebuilt_data.py  ${DATA_DIR}/wsj.${sections}.txt.dep.pp ${DATA_DIR_REBUILT}/wsj.${sections}.txt.dep.pp
    done
}

mk_local_copy

mkdata1


mkdata2
