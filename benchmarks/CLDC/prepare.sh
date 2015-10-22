#!/usr/bin/env bash

tmp=`mktemp -d`
filename=$1
lang=$2

ext="${filename##*.}"
corpus="${filename%.*}"

scripts/prepare.sh $filename $lang | scripts/replace-digits.py > $tmp/tempfile
scripts/reduce-voc.py $tmp/tempfile > $corpus.cldc.$ext
