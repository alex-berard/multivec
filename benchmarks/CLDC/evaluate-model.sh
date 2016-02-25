this_dir=`pwd`
temp_dir=`mktemp -d`

if [ $# -lt 2 ]
  then
    echo "Usage: ./cldc-evaluate.sh MODEL_FILE RCV_DIR"
fi

model=$1
rcv_dir=$2

echo "Output dir: $temp_dir"
./bin/multivec-bi --load $model --save-src $temp_dir/src-model.bin --save-trg $temp_dir/trg-model.bin > /dev/null
./bin/multivec-mono --load $temp_dir/src-model.bin --save-vectors $temp_dir/src-model.txt > /dev/null
./bin/multivec-mono --load $temp_dir/trg-model.bin --save-vectors $temp_dir/trg-model.txt > /dev/null

echo [Bilingual Word Embeddings]

rm -f $rcv_dir/data/embeddings/my-embeddings-de-en.en $rcv_dir/data/embeddings/my-embeddings-de-en.de
rm -f $rcv_dir/data/doc-reprs/*
benchmarks/CLDC/convert-embeddings.py < $temp_dir/src-model.txt > $rcv_dir/data/embeddings/my-embeddings-de-en.en
benchmarks/CLDC/convert-embeddings.py < $temp_dir/trg-model.txt > $rcv_dir/data/embeddings/my-embeddings-de-en.de

cd $rcv_dir/scripts/de2en/
./prepare-data-klement-4cat-all-sizes-my-embeddings.ch > /dev/null
echo ""
echo "DE->EN:"
echo "-----------"
./run-perceptron-all-sizes-my-embeddings.ch
cd ../en2de/
./prepare-data-klement-4cat-all-sizes-my-embeddings.ch > /dev/null
echo ""
echo "EN->DE:"
echo "-----------"
./run-perceptron-all-sizes-my-embeddings.ch

cd $this_dir

echo [Online Paragraph Vector]

rm -f $rcv_dir/data/doc-reprs/*
rm -f $rcv_dir/data/embeddings/my-model.en.bin $rcv_dir/data/embeddings/my-model.de.bin

cp $temp_dir/src-model.bin $rcv_dir/data/embeddings/my-model.en.bin
cp $temp_dir/trg-model.bin $rcv_dir/data/embeddings/my-model.de.bin

cd $rcv_dir/scripts/de2en/
#./prepare-data-klement-4cat-all-sizes-sent-embeddings.ch > /dev/null
./prepare-data-klement-4cat-all-sizes-sent-embeddings.ch
echo ""
echo "DE->EN:"
echo "-----------"
./run-perceptron-all-sizes-my-embeddings.ch
cd ../en2de/
#./prepare-data-klement-4cat-all-sizes-sent-embeddings.ch > /dev/null
./prepare-data-klement-4cat-all-sizes-sent-embeddings.ch
echo ""
echo "EN->DE:"
echo "-----------"
./run-perceptron-all-sizes-my-embeddings.ch

cd $this_dir

