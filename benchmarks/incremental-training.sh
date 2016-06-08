#!/bin/bash

data_dir=data/news-commentary
model_dir=models/news-commentary
corpus=${data_dir}/news
question_file=word2vec/questions-words.txt

mkdir -p ${data_dir}
mkdir -p ${model_dir}

wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz -O ${data_dir}/training-parallel-nc-v9.tgz
tar xzf ${data_dir}/training-parallel-nc-v9.tgz -C ${data_dir}
mv ${data_dir}/training/news-commentary-v9.fr-en.fr ${corpus}.fr
mv ${data_dir}/training/news-commentary-v9.fr-en.en ${corpus}.en
rm -rf ${data_dir}/training
scripts/prepare-data.py ${corpus} ${corpus}.tok fr en --normalize-punk --shuffle

# split in half
head -n 100000 ${corpus}.tok.fr > ${corpus}.1.tok.fr
tail -n -100000 ${corpus}.tok.fr > ${corpus}.2.tok.fr
head -n 100000 ${corpus}.tok.en > ${corpus}.1.tok.en
tail -n -100000 ${corpus}.tok.en > ${corpus}.2.tok.en

voc_size=0

echo "#### Monolingual"

echo "## Baseline"
bin/multivec-mono --train ${corpus}.tok.en --save ${model_dir}/model.en.bin --iter 8 --save-vectors ${model_dir}/vectors.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.en.txt ${voc_size} < ${question_file}

echo "## First half"
bin/multivec-mono --train ${corpus}.1.tok.en --save ${model_dir}/model.1.en.bin --iter 8 --save-vectors ${model_dir}/vectors.1.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.1.en.txt ${voc_size} < ${question_file}

echo "## Second half"
bin/multivec-mono --train ${corpus}.2.tok.en --save ${model_dir}/model.2.en.bin --iter 8 --save-vectors ${model_dir}/vectors.2.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.2.en.txt ${voc_size} < ${question_file}

echo "## First half then second half"
bin/multivec-mono --load ${model_dir}/model.1.en.bin --train ${corpus}.2.tok.en --save ${model_dir}/model.1+2.en.bin --iter 8 --save-vectors ${model_dir}/vectors.1+2.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.1+2.en.txt ${voc_size} < ${question_file}

echo "## Four epochs"
bin/multivec-mono --train ${corpus}.tok.en --save ${model_dir}/model.iter4.en.bin --iter 4 --save-vectors ${model_dir}/vectors.iter4.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.iter4.en.txt ${voc_size} < ${question_file}

echo "## Four epochs then four epochs"
bin/multivec-mono --load ${model_dir}/model.iter4.en.bin --train ${corpus}.tok.en --save ${model_dir}/model.iter4+4.en.bin --iter 4 --save-vectors ${model_dir}/vectors.iter4+4.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.iter4+4.en.txt ${voc_size} < ${question_file}

echo "#### Bilingual"

echo "## Baseline"
bin/multivec-bi --train-src ${corpus}.tok.fr --train-trg ${corpus}.tok.en --save ${model_dir}/model.fr-en.bin --iter 4 --save-trg ${model_dir}/model.fr-en.en.bin --threads 16
bin/multivec-mono --load ${model_dir}/model.fr-en.en.bin --save-vectors ${model_dir}/vectors.fr-en.en.txt
bin/compute-accuracy ${model_dir}/vectors.fr-en.en.txt ${voc_size} < ${question_file}

echo "## First half"
bin/multivec-bi --train-src ${corpus}.1.tok.fr --train-trg ${corpus}.1.tok.en --save ${model_dir}/model.1.fr-en.bin --iter 4 --save-trg ${model_dir}/model.1.fr-en.en.bin --threads 16
bin/multivec-mono --load ${model_dir}/model.1.fr-en.en.bin --save-vectors ${model_dir}/vectors.1.fr-en.en.txt
bin/compute-accuracy ${model_dir}/vectors.1.fr-en.en.txt ${voc_size} < ${question_file}

echo "## Second half"
bin/multivec-bi --train-src ${corpus}.2.tok.fr --train-trg ${corpus}.2.tok.en --save ${model_dir}/model.2.fr-en.bin --iter 4 --save-trg ${model_dir}/model.2.fr-en.en.bin --threads 16
bin/multivec-mono --load ${model_dir}/model.2.fr-en.en.bin --save-vectors ${model_dir}/vectors.2.fr-en.en.txt
bin/compute-accuracy ${model_dir}/vectors.2.fr-en.en.txt ${voc_size} < ${question_file}

echo "## First half then second half"
bin/multivec-bi --load ${model_dir}/model.1.fr-en.bin --train-src ${corpus}.2.tok.fr --train-trg ${corpus}.2.tok.en --save ${model_dir}/model.1+2.fr-en.bin --iter 4 --save-trg ${model_dir}/model.1+2.fr-en.en.bin --threads 16
bin/multivec-mono --load ${model_dir}/model.1+2.fr-en.en.bin --save-vectors ${model_dir}/vectors.1+2.fr-en.en.txt
bin/compute-accuracy ${model_dir}/vectors.1+2.fr-en.en.txt ${voc_size} < ${question_file}

echo "## Two epochs"
bin/multivec-mono --train ${corpus}.tok.en --save ${model_dir}/model.iter4.en.bin --iter 4 --save-vectors ${model_dir}/vectors.iter4.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.iter4.en.txt ${voc_size} < ${question_file}

echo "## Two epochs then four epochs"
bin/multivec-mono --load ${model_dir}/model.iter4.en.bin --train ${corpus}.tok.en --save ${model_dir}/model.iter4+4.en.bin --iter 4 --save-vectors ${model_dir}/vectors.iter4+4.en.txt --threads 16
bin/compute-accuracy ${model_dir}/vectors.iter4+4.en.txt ${voc_size} < ${question_file}

