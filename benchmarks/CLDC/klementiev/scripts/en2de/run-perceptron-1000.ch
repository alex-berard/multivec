datapath=../../data

echo "Training on EN1000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train-my-embeddings.de-EN1000.en  --model-name $datapath/classifiers/avperc.de-en.de   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test-my-embeddings.de-en.de  --model-name $datapath/classifiers/avperc.de-en.de 

