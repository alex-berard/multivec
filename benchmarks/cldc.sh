#!/usr/bin/env bash
corpus=data/europarl/europarl.cldc
output_dir=benchmarks/CLDC/output
threads=16

if [ $# -lt 1 ]
  then
    echo "Usage: ./cldc-evaluate.sh RCV_DIR"
fi

rcv_dir=$1

mkdir -p $output_dir

./bin/multivec-bi --train-src $corpus.en --train-trg $corpus.de --sg --iter 10 --subsampling 1e-04 --alpha 0.025 --beta 4 --dimension 40 --negative 30 --window-size 5 --threads $threads --save $output_dir/model.40.en-de.bin --min-count 1 > $output_dir/model.40.en-de.out

for i in `seq 1 10`;
do
    ./benchmarks/CLDC/evaluate-model.sh $output_dir/model.40.en-de.bin $rcv_dir >> $output_dir/model.40.en-de.out
done

./bin/multivec-bi --train-src $corpus.en --train-trg $corpus.de --sg --iter 10 --subsampling 1e-04 --alpha 0.025 --beta 4 --dimension 128 --negative 30 --window-size 5 --threads $threads --save $output_dir/model.128.en-de.bin --min-count 1 > $output_dir/model.128.en-de.out

for i in `seq 1 10`;  # average over 10 runs
do
    ./benchmarks/CLDC/evaluate-model.sh $output_dir/model.128.en-de.bin $rcv_dir >> $output_dir/model.128.en-de.out
done
