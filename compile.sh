#!/usr/bin/env bash

mkdir -p build
cd build
cmake .. &>/dev/null
make
cd ..
