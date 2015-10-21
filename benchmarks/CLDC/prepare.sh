#!/usr/bin/env bash

tmp=`mktemp -d`
threads=8

filename=$1
lang=$2

ext="${filename##*.}"
corpus="${filename%.*}"

scripts/normalize-punctuation.perl -l $lang < $filename |
scripts/tokenizer.perl -l $lang -threads $threads |
scripts/lowercase.perl |
scripts/replace-digits.py
#scripts/replace-digits.py > $tmp/tempfile
#scripts/reduce-voc.py $tmp/tempfile

#mv $tmp/tempfile $corpus.cldc.$ext
