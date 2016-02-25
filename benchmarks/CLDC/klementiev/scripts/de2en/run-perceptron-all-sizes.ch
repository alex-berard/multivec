datapath=../../data

echo "Training on DE100"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.en-DE100.de  --model-name $datapath/classifiers/avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.en-de.en  --model-name $datapath/classifiers/avperc.en-de.en 

echo "Training on DE200"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.en-DE200.de  --model-name $datapath/classifiers/avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.en-de.en  --model-name $datapath/classifiers/avperc.en-de.en 

echo "Training on DE500"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.en-DE500.de  --model-name $datapath/classifiers/avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.en-de.en  --model-name $datapath/classifiers/avperc.en-de.en 

echo "Training on DE1000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.en-DE1000.de  --model-name $datapath/classifiers/avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.en-de.en  --model-name $datapath/classifiers/avperc.en-de.en 

echo "Training on DE5000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.en-DE5000.de  --model-name $datapath/classifiers/avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.en-de.en  --model-name $datapath/classifiers/avperc.en-de.en 

echo "Training on DE10000"
java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.en-DE10000.de  --model-name $datapath/classifiers/avperc.en-de.en   --epoch-num 10
java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.en-de.en  --model-name $datapath/classifiers/avperc.en-de.en 

