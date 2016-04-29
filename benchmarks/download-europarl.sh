#!/usr/bin/env bash
this_dir=`pwd`   # run this script from project root
data_dir=data/europarl
scripts=$this_dir/scripts
threads=16

mkdir -p $data_dir

cd $data_dir

if [ ! -f europarl.en ]; then
    echo "downloading europarl"
    wget http://www.statmt.org/europarl/v7/de-en.tgz
    tar xzf de-en.tgz
    rm de-en.tgz

    mv europarl-v7.de-en.en europarl.en
    mv europarl-v7.de-en.de europarl.de
fi

if [ ! -f europarl.tok.en ]; then
    echo "pre-processing europarl"
    $scripts/prepare-data.py europarl europarl.tok en de --normalize-punk --tokenize --lowercase --shuffle --threads $threads --scripts $scripts
fi

if [ ! -f europarl.cldc.en ]; then
    echo "pre-processing europarl for CLDC"
    $scripts/prepare-data.py europarl.tok europarl.cldc en de --normalize-numbers --scripts $scripts --min-count 5
fi

cd $this_dir

