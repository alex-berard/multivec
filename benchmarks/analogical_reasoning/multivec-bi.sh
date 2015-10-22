#!/bin/bash

# run from the root directory of the project
filename1=`mktemp`
filename2=`mktemp`

echo "Output file: $filename1"
bin/bivec --save-src $filename1 $@
bin/word2vecpp --load $filename1 --save-vectors-bin $filename2
bin/compute-accuracy $filename2 0 < word2vec/questions-words.txt | tail -n3 | head -n2
