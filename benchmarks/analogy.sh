#!/usr/bin/env bash
corpus=data/europarl/europarl.tok.en
corpus_trg=data/europarl/europarl.tok.de
output=benchmarks/analogical_reasoning/results
threads=16

./benchmarks/download-europarl.sh

mkdir -p $output/word2vec $output/multivec $output/multivec-bi

./benchmarks/analogical_reasoning/multivec.sh --train $corpus --iter 20 --dimension 100 --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec/res1.txt
./benchmarks/analogical_reasoning/multivec.sh --train $corpus --iter 20 --dimension 300 --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec/res2.txt
./benchmarks/analogical_reasoning/multivec.sh --train $corpus --iter 20 --dimension 100 --sg --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec/res3.txt
./benchmarks/analogical_reasoning/multivec.sh --train $corpus --iter 20 --dimension 300 --sg --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec/res4.txt

./benchmarks/analogical_reasoning/word2vec.sh --train $corpus --iter 20 --dimension 100 --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/word2vec/res1.txt
./benchmarks/analogical_reasoning/word2vec.sh --train $corpus --iter 20 --dimension 300 --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/word2vec/res2.txt
./benchmarks/analogical_reasoning/word2vec.sh --train $corpus --iter 20 --dimension 100 --sg --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/word2vec/res3.txt
./benchmarks/analogical_reasoning/word2vec.sh --train $corpus --iter 20 --dimension 300 --sg --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/word2vec/res4.txt

./benchmarks/analogical_reasoning/multivec-bi.sh --train $corpus $corpus_trg --iter 10 --dimension 100 --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec-bi/res1.txt
./benchmarks/analogical_reasoning/multivec-bi.sh --train $corpus $corpus_trg --iter 10 --dimension 300 --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec-bi/res2.txt
./benchmarks/analogical_reasoning/multivec-bi.sh --train $corpus $corpus_trg --iter 10 --dimension 100 --sg --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec-bi/res3.txt
./benchmarks/analogical_reasoning/multivec-bi.sh --train $corpus $corpus_trg --iter 10 --dimension 300 --sg --subsampling 1e-04 --window-size 5 --negative 5 --min-count 5 --threads $threads > $output/multivec-bi/res4.txt
