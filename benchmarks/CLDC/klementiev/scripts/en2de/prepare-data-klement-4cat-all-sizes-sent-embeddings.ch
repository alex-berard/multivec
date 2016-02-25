datapath=../../data
srcpath=../../src

echo [Preparing test set for DE]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/test/de --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/test-my-embeddings.de-en.de

echo [Preparing train set for EN]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN100 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN100.en
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN200 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN200.en
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN500 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN500.en
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN1000 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN1000.en
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN5000 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN5000.en
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN10000 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN10000.en

