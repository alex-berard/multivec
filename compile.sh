#!/usr/bin/env bash

cat CMakeLists.txt | sed "s/add_definitions(.*)/add_definitions($2)/" > CMakeLists.tmp.tmp

cp CMakeLists.txt CMakeLists.tmp
mv CMakeLists.tmp.tmp CMakeLists.txt

cat CMakeLists.txt | grep "add_definitions"

mkdir -p build
cd build
cmake .. &>/dev/null
make #&>/dev/null
cd ..
if [ "$1" != "" ] && [ "$1" != "bin/multivec-mono" ]
then
    echo $1
    cp bin/multivec-mono $1
fi

mv CMakeLists.tmp CMakeLists.txt
