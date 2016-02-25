datapath=../../data

echo [Preparing test set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/test/EN1000_train_valid --idf $datapath/idfs/idf.en --word-embeddings $datapath/embeddings/my-embeddings-de-en.en  --vector-file $datapath/doc-reprs/test-my-embeddings.EN1000_train_valid.en

echo [Preparing train set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/EN1000_train_valid --idf $datapath/idfs/idf.en --word-embeddings $datapath/embeddings/my-embeddings-de-en.en  --vector-file $datapath/doc-reprs/train-my-embeddings.EN1000_train_valid.en
