#!/bin/bash

# run from the root directory of the project
filename=`mktemp`
word2vec/word2vec --train data/news.en --save-embeddings $filename $@
word2vec/compute-accuracy $filename 0 < data/questions-words.txt | tail -n3 | head -n2
