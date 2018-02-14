#!/usr/bin/env bash

set -e

output=benchmarks/analogy.log
threads=16

function word2vec {
    filename=`mktemp`
    command="bin/word2vec $@ --save-vectors ${filename}"
    echo ${command}
    ${command} | grep "Training time"
    bin/analogy ${filename} word2vec/questions-words.txt | tail -n4
    rm -f ${filename}
}

function multivec {
    filename=`mktemp`
    command="bin/multivec $@ --save-vectors ${filename}"
    echo ${command}
    ${command} | grep "Training time"
    bin/analogy ${filename} word2vec/questions-words.txt | tail -n4
    rm -f ${filename}
}

function multivec-bi {
    filename=`mktemp`
    command="bin/multivec-bi $@ --save-src-vectors ${filename}"
    echo ${command}
    ${command}| grep "Training time"
    bin/analogy ${filename} word2vec/questions-words.txt | tail -n4
    rm -f ${filename}
}

rm -f ${output}
for hs in "--negative 10" "--hs --negative 0"
do
    for corpus in news news+europarl news+europarl+commoncrawl
    do
        iter=20
        bi_iter=10

        if [ ${corpus} = news ]
        then
            iter=40
            bi_iter=20
        fi

        params="--subsampling 1e-04 --window-size 5 --min-count 5 --threads ${threads}"
        params_mono="--train data/${corpus}.en --iter ${iter} ${params} "
        params_bi="--train-src data/${corpus}.en --train-trg data/${corpus}.de --iter ${bi_iter} ${params}"

        multivec ${params_mono} --dimension 100 --alpha 0.1 ${hs} >> ${output}
        multivec ${params_mono} --dimension 300 --alpha 0.1 ${hs} >> ${output}
        multivec ${params_mono} --dimension 100 --sg ${hs} >> ${output}
        multivec ${params_mono} --dimension 300 --sg ${hs} >> ${output}

        word2vec ${params_mono} --dimension 100 --alpha 0.1 ${hs} >> ${output}
        word2vec ${params_mono} --dimension 300 --alpha 0.1 ${hs} >> ${output}
        word2vec ${params_mono} --dimension 100 --sg ${hs} >> ${output}
        word2vec ${params_mono} --dimension 300 --sg ${hs} >> ${output}

        multivec-bi ${params_bi} --dimension 100 --alpha 0.1 ${hs} >> ${output}
        multivec-bi ${params_bi} --dimension 300 --alpha 0.1 ${hs} >> ${output}
        multivec-bi ${params_bi} --dimension 100 --sg ${hs} >> ${output}
        multivec-bi ${params_bi} --dimension 300 --sg ${hs} >> ${output}

    done
done