#!/usr/bin/env bash
# run this script from project root
raw_data=raw_data
data_dir=data
mkdir -p ${raw_data}
mkdir -p ${data_dir}

wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip -P ${raw_data}
unzip ${raw_data}/stanfordSentimentTreebank.zip -d ${raw_data}

