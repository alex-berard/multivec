#!/bin/bash

set -e

ext_corpus=data/news+europarl+commoncrawl.en
output=benchmarks/sentiment.log
data_dir=benchmarks/imdb_sentiment/data
liblinear=benchmarks/imdb_sentiment/liblinear-2.20
threads=16

tmp_dir=`mktemp -d`

params="--iter 40 --dimension 100 --subsampling 1e-04 --window-size 10 --negative 15 --threads ${threads} --min-count 1 --alpha 0.1"
reco_params="--iter 20 --sg --hs 1 --dimension 100 --subsampling 1e-3 --window-size 10 --negative 5 --threads ${threads} --min-count 1"
ext_iter="10"
online_iter="40"

function eval {
    head ${tmp_dir}/sentence_vectors.txt -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > ${tmp_dir}/train.txt
    head ${tmp_dir}/sentence_vectors.txt -n 50000 | tail -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > ${tmp_dir}/test.txt
    ${liblinear}/train -s 0 ${tmp_dir}/train.txt ${tmp_dir}/model.logreg -q &>/dev/null
    ${liblinear}/predict ${tmp_dir}/test.txt ${tmp_dir}/model.logreg /dev/null
}

if ! [ -f ${data_dir}/alldata.shuf.txt ]
then
    benchmarks/imdb_sentiment/prepare.sh
fi

cat ${ext_corpus} ${data_dir}/alldata.shuf.txt ${ext_corpus} | shuf > ${tmp_dir}/ext+train+test.en
cat ${ext_corpus} ${data_dir}/train.shuf.txt ${ext_corpus} | shuf > ${tmp_dir}/ext+train.en

rm -f ${output}

echo "## word2vec batch PV" >> ${output}
command="bin/word2vec --train ${data_dir}/alldata-id.shuf.txt --save-vectors ${tmp_dir}/vectors.txt --sent-vector ${params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
grep '_\*' ${tmp_dir}/vectors.txt | sed s/^..// | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## word2vec batch PV - recommended" >> ${output}
command="bin/word2vec --train ${data_dir}/alldata-id.shuf.txt --save-vectors ${tmp_dir}/vectors.txt --sent-vector ${reco_params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
grep '_\*' ${tmp_dir}/vectors.txt | sed s/^..// | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## word2vec batch PV - unshuf" >> ${output}
command="bin/word2vec --train ${data_dir}/alldata-id.txt --save-vectors ${tmp_dir}/vectors.txt --sent-vector ${reco_params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
grep '_\*' ${tmp_dir}/vectors.txt | sed s/^..// > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec batch PV" >> ${output}
command="bin/multivec --train ${data_dir}/alldata.shuf.txt --save-sent-vectors ${tmp_dir}/vectors.txt --sent-vector ${params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## multivec batch PV - unshuf" >> ${output}
command="bin/multivec --train ${data_dir}/alldata.txt --save-vectors ${tmp_dir}/vectors.txt --sent-vector ${params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
grep '_\*' ${tmp_dir}/vectors.txt | sed s/^..// > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## multivec batch PV - unshuf HS" >> ${output}
command="bin/multivec --train ${data_dir}/alldata.txt --save-vectors ${tmp_dir}/vectors.txt --sent-vector ${params} --hs"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
grep '_\*' ${tmp_dir}/vectors.txt | sed s/^..// > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## word2vec BOW" >> ${output}
command="bin/word2vec --train ${data_dir}/train.shuf.txt --save-vectors ${tmp_dir}/word-vectors.txt ${params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
scripts/vector-sum.py ${tmp_dir}/word-vectors.txt ${data_dir}/alldata.shuf.txt --avg > ${tmp_dir}/vectors.txt
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec BOW" >> ${output}
command="bin/multivec --train ${data_dir}/train.shuf.txt --save-vectors ${tmp_dir}/word-vectors.txt ${params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
scripts/vector-sum.py ${tmp_dir}/word-vectors.txt ${data_dir}/alldata.shuf.txt --avg > ${tmp_dir}/vectors.txt
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec online PV" >> ${output}
command="bin/multivec --train ${data_dir}/train.shuf.txt --sent-vector ${params} --save ${tmp_dir}/model.bin"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
bin/multivec --load ${tmp_dir}/model.bin ${params} --online-sent-vector ${data_dir}/alldata.shuf.txt --save-sent-vectors ${tmp_dir}/vectors.txt --iter ${online_iter} &>/dev/null
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec online PV ext corpus" >> ${output}
command="bin/multivec --train ${ext_corpus} --sent-vector ${params} --iter ${ext_iter} --save ${tmp_dir}/model.bin"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
bin/multivec --load ${tmp_dir}/model.bin ${params} --online-sent-vector ${data_dir}/alldata.shuf.txt --save-sent-vectors ${tmp_dir}/vectors.txt --iter ${online_iter} &>/dev/null
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec BOW ext corpus" >> ${output}
command="bin/multivec --train ${ext_corpus} --save-vectors ${tmp_dir}/word-vectors.txt ${params} --iter ${ext_iter}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
scripts/vector-sum.py ${tmp_dir}/word-vectors.txt ${data_dir}/alldata.shuf.txt --avg > ${tmp_dir}/vectors.txt
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec BOW train + test" >> ${output}
command="bin/multivec --train ${data_dir}/alldata.shuf.txt --save-vectors ${tmp_dir}/word-vectors.txt ${params}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
scripts/vector-sum.py ${tmp_dir}/word-vectors.txt ${data_dir}/alldata.shuf.txt --avg > ${tmp_dir}/vectors.txt
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec online PV ext corpus + train" >> ${output}
command="bin/multivec --train ${tmp_dir}/ext+train.en --sent-vector ${params} --iter ${ext_iter} --save ${tmp_dir}/model.bin"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
bin/multivec --load ${tmp_dir}/model.bin ${params} --online-sent-vector ${data_dir}/alldata.shuf.txt --save-sent-vectors ${tmp_dir}/vectors.txt --iter ${online_iter} &>/dev/null
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec BOW ext corpus + train + test" >> ${output}
command="bin/multivec --train ${tmp_dir}/ext+train+test.en --save-vectors ${tmp_dir}/word-vectors.txt ${params} --iter ${ext_iter}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
scripts/vector-sum.py ${tmp_dir}/word-vectors.txt ${data_dir}/alldata.shuf.txt --avg > ${tmp_dir}/vectors.txt
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

echo "## MultiVec BOW ext corpus + train" >> ${output}
command="bin/multivec --train ${tmp_dir}/ext+train.en --save-vectors ${tmp_dir}/word-vectors.txt ${params} --iter ${ext_iter}"
echo ${command} >> ${output}
${command} | grep "Training time" >> ${output}
scripts/vector-sum.py ${tmp_dir}/word-vectors.txt ${data_dir}/alldata.shuf.txt --avg > ${tmp_dir}/vectors.txt
paste -d " " ${data_dir}/just-ids.shuf.txt ${tmp_dir}/vectors.txt | sort -nk 1,1 > ${tmp_dir}/sentence_vectors.txt
eval >> ${output}

rm -rf ${tmp_dir}
