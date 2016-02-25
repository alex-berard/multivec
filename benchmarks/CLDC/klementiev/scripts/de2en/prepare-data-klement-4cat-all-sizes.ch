datapath=../../data

echo [Preparing test set for EN]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/test/en --idf $datapath/idfs/idf.en --word-embeddings $datapath/embeddings/original-de-en.en  --vector-file $datapath/doc-reprs/test.en-de.en

echo [Preparing train set for DE]
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE100 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.en-DE100.de
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE200 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.en-DE200.de
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE500 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.en-DE500.de
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE1000 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.en-DE1000.de
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE5000 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.en-DE5000.de
java -ea -Xmx2000m  -cp ../../bin CollectionPreprocessor --text-dir $datapath/rcv-from-binod/train/DE10000 --idf $datapath/idfs/idf.de --word-embeddings $datapath/embeddings/original-de-en.de  --vector-file $datapath/doc-reprs/train.en-DE10000.de
