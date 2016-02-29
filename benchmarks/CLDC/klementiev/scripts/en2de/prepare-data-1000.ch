datapath=../../data

echo [Preparing test set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/test/de --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/my-embeddings-de-en.de  --vector-file $datapath/doc-reprs/test-my-embeddings.de-en.de

echo [Preparing train set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/EN1000 --idf $datapath/idfs/idf.en --word-embeddings $datapath/embeddings/my-embeddings-de-en.en  --vector-file $datapath/doc-reprs/train-my-embeddings.de-EN1000.en

