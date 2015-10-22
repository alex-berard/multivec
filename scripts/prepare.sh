#!/usr/bin/env bash

threads=8
filename=$1
lang=$2

ext="${filename##*.}"
corpus="${filename%.*}"

scripts/normalize-punctuation.perl -l $lang < $filename |
scripts/tokenizer.perl -l $lang -threads $threads |
scripts/lowercase.perl
