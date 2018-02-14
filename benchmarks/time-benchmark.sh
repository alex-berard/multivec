#!/usr/bin/env bash

corpus=$1
threads=$2
iter=10
alpha=0.1
subsampling=1e-03
negative=10
window_size=5
min_count=5
dim=100
sg=
cos_mul=

runs=10

params="--train data/${corpus}.en ${sg} --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim ${dim}"

for bin in bin/word2vec bin/multivec bin/multivec-v{1,2,3,4,5,6}
do
    log_filename=`mktemp`
    vec_filename=`mktemp`
    
    printf "Binary: %s\nLog file: %s\nVectors: %s\n" ${bin} ${log_filename} ${vec_filename}
    
    for i in `seq ${runs}`
    do
        rm -f ${vec_filename}
        ${bin} ${params} --save-vectors ${vec_filename} >> ${log_filename}
        bin/analogy ${vec_filename} word2vec/questions-words.txt ${cos_mul} >> ${log_filename}
    done
    
    total_time=0
    accuracy=0
    
    for x in `cat ${log_filename} | grep -Po "Training time:\s*\K\d+\.?\d*"`
    do
        total_time=`echo ${total_time} + $x | bc`
    done
    for x in `cat ${log_filename} | grep -Po "Total accuracy:\s*\K\d+\.?\d*"`
    do
        accuracy=`echo ${accuracy} + $x | bc`
    done
    
    total_time=`echo "scale=1; ${total_time} / ${runs}.0" | bc`
    accuracy=`echo "scale=1; ${accuracy} / ${runs}.0" | bc`
    
    printf "Time: %s sec, Accuracy: %s%%\n" ${total_time} ${accuracy}
done
