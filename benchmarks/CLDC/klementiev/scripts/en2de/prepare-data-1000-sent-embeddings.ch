datapath=../../data
srcpath=../../src

echo [Preparing test set for DE]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/test/de --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/test-my-embeddings.de-en.de

echo [Preparing train set for EN]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/EN1000 --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/train-my-embeddings.de-EN1000.en

