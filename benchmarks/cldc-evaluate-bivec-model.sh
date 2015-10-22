this_dir=`pwd`
temp_dir=`mktemp -d`
model=$1
klementiev=../document-representations

echo "Output dir: $temp_dir"
./bin/bivec --load $model --save-src $temp_dir/src-model.bin --save-trg $temp_dir/trg-model.bin > /dev/null
./bin/word2vecpp --load $temp_dir/src-model.bin --save-vectors $temp_dir/src-model.txt > /dev/null
./bin/word2vecpp --load $temp_dir/trg-model.bin --save-vectors $temp_dir/trg-model.txt > /dev/null

rm -f $klementiev/data/embeddings/my-embeddings-de-en.en $klementiev/data/embeddings/my-embeddings-de-en.de
rm -f $klementiev/data/doc-reprs/*
benchmarks/CLDC/convert-embeddings.py < $temp_dir/src-model.txt > $klementiev/data/embeddings/my-embeddings-de-en.en
benchmarks/CLDC/convert-embeddings.py < $temp_dir/trg-model.txt > $klementiev/data/embeddings/my-embeddings-de-en.de

cd $klementiev/scripts/de2en/
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
