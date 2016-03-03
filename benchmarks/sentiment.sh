#!/bin/bash
ext_corpus=data/europarl/europarl.tok.en
output=benchmarks/imdb_sentiment/results
threads=16

rm -rf $output
mkdir -p $output

./benchmarks/imdb_sentiment/word2vec-sum.sh --iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads $threads > $output/res1.txt
./benchmarks/imdb_sentiment/word2vec-sum-ext-corpus.sh --train $ext_corpus --iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads $threads > $output/res2.txt
./benchmarks/imdb_sentiment/word2vec.sh --iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads $threads > $output/res3.txt
./benchmarks/imdb_sentiment/multivec.sh --iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads $threads > $output/res4.txt
./benchmarks/imdb_sentiment/multivec-online.sh --iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads $threads > $output/res5.txt
./benchmarks/imdb_sentiment/multivec-ext-corpus.sh --train $ext_corpus --iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads $threads > $output/res6.txt
