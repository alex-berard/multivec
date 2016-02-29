datapath=../../data

echo "Training on DE1000"
java  -ea -Xmx2000m -cp ../../bin ApLearn --train-set  $datapath/doc-reprs/train-my-embeddings.en-DE1000.de --model-name $datapath/classifiers/avperc.en-de.en --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin ApClassify --test-set $datapath/doc-reprs/test-my-embeddings.en-de.en --model-name $datapath/classifiers/avperc.en-de.en
