#!/bin/bash

# run from the root directory of the project
filename1=`mktemp`
filename2=`mktemp`

echo "Output file: $filename1"
bin/multivec-bi --save-src $filename1 $@
bin/multivec-mono --load $filename1 --save-vectors-bin $filename2
bin/compute-accuracy $filename2 0 < word2vec/questions-words.txt | tail -n3 | head -n2
