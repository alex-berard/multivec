#!/bin/bash

# run from the root directory of the project
filename=`mktemp`
echo "Output file: $filename"
bin/word2vec --save-vectors $filename $@
bin/compute-accuracy $filename word2vec/questions-words.txt | tail -n5
