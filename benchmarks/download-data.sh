#!/usr/bin/env bash
# run this script from project root
raw_data=raw_data
data_dir=data
mkdir -p ${raw_data}
mkdir -p ${data_dir}

function process {
    scripts/prepare-data.py ${1} ${2} en de --normalize-punk --tokenize --lowercase --shuffle --min 1
}
function process-cldc {
    scripts/prepare-data.py ${1} ${2} en de --normalize-digits --min-count 5 --min 1
}

if [ ! -f ${raw_data}/training-parallel-nc-v9.tgz ]
then
    wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz -P ${raw_data}
fi
tar xzf ${raw_data}/training-parallel-nc-v9.tgz -C ${raw_data}
process ${raw_data}/training/news-commentary-v9.de-en ${data_dir}/news
scripts/shuf-corpus.py ${data_dir}/news en de
rm -rf ${raw_data}/training

if [ ! -f ${raw_data}/training-parallel-europarl-v7.tgz ]
then
    wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz -P ${raw_data}
fi
tar xzf ${raw_data}/training-parallel-europarl-v7.tgz -C ${raw_data}
process ${raw_data}/training/europarl-v7.de-en ${data_dir}/europarl
process-cldc ${data_dir}/europarl ${data_dir}/europarl.cldc
cat ${data_dir}/{news,europarl}.en >> ${data_dir}/news+europarl.en
cat ${data_dir}/{news,europarl}.de >> ${data_dir}/news+europarl.de
scripts/shuf-corpus.py ${data_dir}/news+europarl en de
rm -rf ${raw_data}/training

if [ ! -f ${raw_data}/training-parallel-commoncrawl.tgz ]
then
    wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz -P ${raw_data}
fi
tar xzf ${raw_data}/training-parallel-commoncrawl.tgz -C ${raw_data}
process ${raw_data}/commoncrawl.de-en ${data_dir}/commoncrawl
cat ${data_dir}/{news,europarl,commoncrawl}.en >> ${data_dir}/news+europarl+commoncrawl.en
cat ${data_dir}/{news,europarl,commoncrawl}.de >> ${data_dir}/news+europarl+commoncrawl.de
scripts/shuf-corpus.py ${data_dir}/news+europarl+commoncrawl en de
rm -f ${raw_data}/commoncrawl.*
