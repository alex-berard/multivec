datapath=../../data

echo [Preparing test set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/test/DE1000_train_valid --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/test.DE1000_train_valid.de

echo [Preparing train set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE1000_train_valid --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.DE1000_train_valid.de
