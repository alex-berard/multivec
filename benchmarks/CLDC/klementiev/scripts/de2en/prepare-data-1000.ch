datapath=../../data

echo [Preparing test set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/test/en --idf $datapath/idfs/idf.en --word-embeddings $datapath/embeddings/my-embeddings-de-en.en --vector-file $datapath/doc-reprs/test-my-embeddings.en-de.en

echo [Preparing train set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE1000 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/my-embeddings-de-en.de --vector-file $datapath/doc-reprs/train-my-embeddings.en-DE1000.de
