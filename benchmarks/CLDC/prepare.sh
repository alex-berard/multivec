#!/usr/bin/env bash
tmp=`mktemp -d`
this_dir=`pwd`
data_dir=data/europarl
scripts=$this_dir/scripts
threads=16

cd $data_dir
wget http://www.statmt.org/europarl/v7/de-en.tgz
tar xzf de-en.tgz
rm de-en.tgz

mv europarl-v7.de-en.en europarl.en
mv europarl-v7.de-en.de europarl.de

cat europarl.en | $scripts/tokenizer.perl -l en -threads $threads | $scripts/lowercase.perl | $scripts/replace-digits.py > $tmp/europarl.cldc.en
$scripts/reduce-voc.py $tmp/europarl.cldc.en > europarl.cldc.en
rm $tmp/europarl.cldc.en

cat europarl.de | $scripts/tokenizer.perl -l de -threads $threads | $scripts/lowercase.perl | $scripts/replace-digits.py > $tmp/europarl.cldc.de
$scripts/reduce-voc.py $tmp/europarl.cldc.de > europarl.cldc.de
rm $tmp/europarl.cldc.de

