#!/bin/bash

# run from the root directory of the project
filename=`mktemp`
bin/word2vec --train data/news.en --save-vectors-bin $filename $@
bin/compute-accuracy $filename 0 < data/questions-words.txt | tail -n3 | head -n2
