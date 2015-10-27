#!/usr/bin/env bash

output_dir=`mktemp -d`
data_dir=benchmarks/imdb_sentiment/data
liblinear=benchmarks/imdb_sentiment/liblinear-2.1
rm -rf $output_dir
mkdir $output_dir

echo "Output directory: $output_dir"

bin/multivec-mono --train $data_dir/alldata.shuf.txt --save-sent-vectors $output_dir/vectors.txt --sent-vector --min-count 1 $@
paste -d " " $data_dir/just-ids.shuf.txt $output_dir/vectors.txt | sort -nk 1,1 > $output_dir/sentence_vectors.txt

head $output_dir/sentence_vectors.txt -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > $output_dir/train.txt
head $output_dir/sentence_vectors.txt -n 50000 | tail -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > $output_dir/test.txt
${liblinear}/train -s 0 $output_dir/train.txt $output_dir/model.logreg
${liblinear}/predict -b 1 $output_dir/test.txt $output_dir/model.logreg $output_dir/out.logreg
tail -n 25000 $output_dir/out.logreg > $output_dir/SENTENCE-VECTOR.LOGREG

cat $output_dir/SENTENCE-VECTOR.LOGREG | awk ' \
BEGIN{cn=0; corr=0;} \
{ \
  if ($2>0.5) if (cn<12500) corr++; \
  if ($2<0.5) if (cn>=12500) corr++; \
  cn++; \
} \
END{print "Sentence vector + logistic regression accuracy: " corr/cn*100 "%";}'
