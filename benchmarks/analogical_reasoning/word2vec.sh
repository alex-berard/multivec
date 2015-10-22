#!/bin/bash

# run from the root directory of the project
filename=`mktemp`
echo "Output file: $filename"
bin/word2vec --save-vectors-bin $filename $@
bin/compute-accuracy $filename 0 < word2vec/questions-words.txt | tail -n3 | head -n2
