#!/usr/bin/env bash
data_dir=data

#this function will convert text to lowercase and will disconnect punctuation and special symbols from words
function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

#wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#tar -xvf aclImdb_v1.tar.gz

for j in train/pos train/neg test/pos test/neg train/unsup; do
  rm -f temp
  for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
  normalize_text temp
  mv temp-norm aclImdb/$j/norm.txt
done

rm -f temp

mv aclImdb/train/pos/norm.txt $data_dir/train-pos.txt
mv aclImdb/train/neg/norm.txt $data_dir/train-neg.txt
mv aclImdb/test/pos/norm.txt $data_dir/test-pos.txt
mv aclImdb/test/neg/norm.txt $data_dir/test-neg.txt
mv aclImdb/train/unsup/norm.txt $data_dir/train-unsup.txt

cat $data_dir/train-pos.txt $data_dir/train-neg.txt $data_dir/test-pos.txt $data_dir/test-neg.txt $data_dir/train-unsup.txt > $data_dir/alldata.txt
awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < $data_dir/alldata.txt > $data_dir/alldata-id.txt
shuf $data_dir/alldata-id.txt > $data_dir/alldata-id.shuf.txt

#mkdir rnnlm
#cd rnnlm
#wget http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-0.3e.tgz
#tar -xvf rnnlm-0.3e.tgz
#g++ -lm -O3 -march=native -Wall -funroll-loops -ffast-math -c rnnlmlib.cpp
#g++ -lm -O3 -march=native -Wall -funroll-loops -ffast-math rnnlm.cpp rnnlmlib.o -o rnnlm

#head ../train-pos.txt -n 12300 > train
#tail ../train-pos.txt -n 200 > valid
#./rnnlm -rnnlm model-pos -train train -valid valid -hidden 50 -direct-order 3 -direct 200 -class 100 -debug 2 -bptt 4 -bptt-block 10 -binary

#head ../train-neg.txt -n 12300 > train
#tail ../train-neg.txt -n 200 > valid
#./rnnlm -rnnlm model-neg -train train -valid valid -hidden 50 -direct-order 3 -direct 200 -class 100 -debug 2 -bptt 4 -bptt-block 10 -binary

#cat ../test-pos.txt ../test-neg.txt > test.txt
#awk 'BEGIN{a=0;}{print a " " $0; a++;}' < test.txt > test-id.txt
#./rnnlm -rnnlm model-pos -test test-id.txt -debug 0 -nbest > model-pos-score
#./rnnlm -rnnlm model-neg -test test-id.txt -debug 0 -nbest > model-neg-score
#paste model-pos-score model-neg-score | awk '{print $1 " " $2 " " $1/$2;}' > ../RNNLM-SCORE
#cd ..

liblinear=liblinear-2.1
wget http://www.csie.ntu.edu.tw/~cjlin/liblinear/${liblinear}.zip
unzip ${liblinear}.zip
cd ${liblinear}
make
cd ..
