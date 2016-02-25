datapath=../../data

java  -ea -Xmx2000m -cp ../../bin ApLearn  --train-set  $datapath/doc-reprs/train.EN1000_train_valid.en  --model-name $datapath/classifiers/avperc.EN1000_train_valid.en   --epoch-num 10

java  -ea -Xmx2000m -cp ../../bin   ApClassify  --test-set $datapath/doc-reprs/test.EN1000_train_valid.en  --model-name $datapath/classifiers/avperc.EN1000_train_valid.en


