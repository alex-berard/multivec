datapath=../../data
srcpath=../../src

echo [Preparing test set for EN]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/test/en --model-file $datapath/embeddings/my-model.en.bin --output-file $datapath/doc-reprs/test-my-embeddings.en-de.en

echo [Preparing train set for DE]
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE100 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE100.de
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE200 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE200.de
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE500 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE500.de
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE1000 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE1000.de
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE5000 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE5000.de
$srcpath/preprocess.py --text-dir $datapath/rcv-from-binod/train/DE10000 --model-file $datapath/embeddings/my-model.de.bin --output-file $datapath/doc-reprs/train-my-embeddings.en-DE10000.de

