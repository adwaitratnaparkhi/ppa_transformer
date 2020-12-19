#!/bin/bash



if [ $# != 3 ]
    then
    echo "usage: $0 [Treebank Dir] [Training output] [Test output]"
    exit 1
fi

TBDIR=$1
TROUT=$2
TEOUT=$3

TR="02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21"
TE="23"

for x in ${TR}
do
    cat ${TBDIR}/${x}/*.mrg
done | 
java -jar  pennconverter.jar -rightBranching=false > ${TROUT}

for x in ${TE}
do
    cat ${TBDIR}/${x}/*.mrg
done | 
    java -jar  pennconverter.jar -rightBranching=false > ${TEOUT}




