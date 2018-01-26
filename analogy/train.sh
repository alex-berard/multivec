#!/usr/bin/env bash
corpus=$1
threads=$2
iter=20
subsampling=1e-04
negative=10
window_size=5
min_count=5

alpha=0.1

bin/multivec-mono --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 100 --save models/${corpus}.cbow100 --save-vectors models/${corpus}.vec > analogy/${corpus}.log
bin/compute-accuracy models/${corpus}.vec word2vec/questions-words.txt >> analogy/${corpus}.log

bin/multivec-mono --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 300 --save models/${corpus}.cbow300 --save-vectors models/${corpus}.vec >> analogy/${corpus}.log
bin/compute-accuracy models/${corpus}.vec word2vec/questions-words.txt >> analogy/${corpus}.log

alpha=0.05

bin/multivec-mono --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 100 --sg --save models/${corpus}.sg100 --save-vectors models/${corpus}.vec >> analogy/${corpus}.log
bin/compute-accuracy models/${corpus}.vec word2vec/questions-words.txt >> analogy/${corpus}.log

bin/multivec-mono --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 300 --sg --save models/${corpus}.sg300 --save-vectors models/${corpus}.vec >> analogy/${corpus}.log
bin/compute-accuracy models/${corpus}.vec word2vec/questions-words.txt >> analogy/${corpus}.log
