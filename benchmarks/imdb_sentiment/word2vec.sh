
output_dir=`mktemp -d`
data_dir=benchmarks/imdb_sentiment/data
liblinear=benchmarks/imdb_sentiment/liblinear-2.1
rm -rf $output_dir
mkdir $output_dir

#cp ../../word2vec/sent2vec.c $output_dir
#gcc $output_dir/sent2vec.c -o $output_dir/word2vec -lm -pthread -O3 -march=native -funroll-loops
cp word2vec/word2vec $output_dir

$output_dir/word2vec --train $data_dir/alldata-id.txt --save-embeddings-txt $output_dir/vectors.txt --sg --dimension 100 --window-size 10 --negative 5 --subsampling 1e-4 --threads 40 --iter 20 --min-count 1 --sent-ids
#time $output_dir/word2vec -train $data_dir/alldata-id.txt -output $output_dir/vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
#time ./word2vec -train ../alldata-id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 1 -sample 1e-3 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1 # LE's version
grep '_\*' $output_dir/vectors.txt | sed s/^..// | sort -nk 1,1 > $output_dir/sentence_vectors.txt

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
