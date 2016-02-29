datapath=../../data
srcpath=../../src

echo [Preparing test set for EN]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/test/en --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/test-my-embeddings.en-de.en

echo [Preparing train set for DE]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE1000 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE1000.de
