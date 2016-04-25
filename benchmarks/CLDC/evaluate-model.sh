#!/usr/bin/env bash
this_dir=`pwd`
temp_dir=`mktemp -d`

if [ $# -lt 2 ]
  then
    echo "Usage: ./cldc-evaluate.sh MODEL_FILE RCV_DIR"
    exit
fi

model=$1
rcv_dir=$2

rm -f $rcv_dir/data/embeddings/*

./bin/multivec-bi --load $model --save-src $temp_dir/src-model.bin --save-trg $temp_dir/trg-model.bin > /dev/null
./bin/multivec-mono --load $temp_dir/src-model.bin --saving-policy 2 --save-vectors $rcv_dir/data/embeddings/my-embeddings-de-en.en > /dev/null
./bin/multivec-mono --load $temp_dir/trg-model.bin --saving-policy 2 --save-vectors $rcv_dir/data/embeddings/my-embeddings-de-en.de > /dev/null

echo [Bilingual Word Embeddings]
cd $rcv_dir/scripts/de2en/
rm $rcv_dir/data/doc-reprs/*
./prepare-data-1000.ch > /dev/null
echo ""
echo "DE->EN:"
echo "-----------"
./run-perceptron-1000.ch
cd ../en2de/
rm $rcv_dir/data/doc-reprs/*
./prepare-data-1000.ch > /dev/null
echo ""
echo "EN->DE:"
echo "-----------"
./run-perceptron-1000.ch

cd $this_dir

echo [Online Paragraph Vector]

cp $temp_dir/src-model.bin $rcv_dir/data/embeddings/my-model.en.bin
cp $temp_dir/trg-model.bin $rcv_dir/data/embeddings/my-model.de.bin

cd $rcv_dir/scripts/de2en/
rm $rcv_dir/data/doc-reprs/*
./prepare-data-1000-sent-embeddings.ch > /dev/null
echo ""
echo "DE->EN:"
echo "-----------"
./run-perceptron-1000.ch
cd ../en2de/
rm $rcv_dir/data/doc-reprs/*
./prepare-data-1000-sent-embeddings.ch > /dev/null
echo ""
echo "EN->DE:"
echo "-----------"
./run-perceptron-1000.ch

cd $this_dir

