#!/usr/bin/env bash
corpus=$1
threads=$2
iter=20
subsampling=1e-04
negative=10
window_size=5
min_count=5

alpha=0.1

bin/word2vec --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 100 --save-vectors models/w2v.${corpus}.cbow100.vec > analogy/w2v.${corpus}.log
bin/compute-accuracy models/w2v.${corpus}.cbow100.vec word2vec/questions-words.txt >> analogy/w2v.${corpus}.log

bin/word2vec --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 300 --save-vectors models/w2v.${corpus}.cbow300.vec >> analogy/w2v.${corpus}.log
bin/compute-accuracy models/w2v.${corpus}.cbow300.vec word2vec/questions-words.txt >> analogy/w2v.${corpus}.log

alpha=0.05

bin/word2vec --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 100 --sg --save-vectors models/w2v.${corpus}.sg100.vec >> analogy/w2v.${corpus}.log
bin/compute-accuracy models/w2v.${corpus}.sg100.vec word2vec/questions-words.txt >> analogy/w2v.${corpus}.log

bin/word2vec --train data/${corpus}.en --threads ${threads} --alpha ${alpha} --iter ${iter} --subsampling ${subsampling} --negative ${negative} --window-size ${window_size} --min-count ${min_count} --dim 300 --sg --save-vectors models/w2v.${corpus}.sg300.vec >> analogy/w2v.${corpus}.log
bin/compute-accuracy models/w2v.${corpus}.sg300.vec word2vec/questions-words.txt >> analogy/w2v.${corpus}.log
