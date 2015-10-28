corpus=data/europarl.cldc
output_dir=benchmarks/CLDC/output
threads=16

mkdir -p $output_dir
./bin/multivec-bi --train-src $corpus.en --train-trg $corpus.de --sg --iter 10 --subsampling 1e-04 --alpha 0.025 --beta 4 --dimension 128 --negative 30 --window-size 5 --threads $threads --save $output_dir/bivec-model.128.en-de.bin --min-count 1 > $output_dir/bivec-model.128.en-de.out

./benchmarks/cldc-evaluate-bivec-model.sh $output_dir/bivec-model.128.en-de.bin >> $output_dir/bivec-model.128.en-de.out

./bin/multivec-bi --train-src $corpus.en --train-trg $corpus.de --sg --iter 10 --subsampling 1e-04 --alpha 0.025 --beta 4 --dimension 40 --negative 30 --window-size 5 --threads $threads --save $output_dir/bivec-model.40.en-de.bin --min-count 1 > $output_dir/bivec-model.40.en-de.out

./benchmarks/cldc-evaluate-bivec-model.sh $output_dir/bivec-model.40.en-de.bin >> $output_dir/bivec-model.40.en-de.out
