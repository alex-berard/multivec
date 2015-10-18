# word2vecpp
C++ implementation of word2vec, bivec, and paragraph vector.

This project is basically the same as the [word2vecpp](https://github.com/eske/word2vecpp) project,
except that it doesn't use Armadillo, and is compatible with GCC 4.4+ (C++0x).

## Dependencies
* Boost.Program_options
* Boost.Serialization

On Ubuntu/Debian:

    apt-get install libboost-program-options-dev libboost-serialization-dev

## TODO
* fix hierarchical softmax for paragraph vector
* fix online paragraph vector
* possibility to reduce vocabulary when training paragraph vector
* program option to save paragraph vectors trained online
* incremental training: possibility to call train without erasing the model (to improve existing model or initialize bilingual model
with monolingual data)
* GIZA alignment for bilingual model
* when using negative sampling, possibility to export as word embeddings either input weights or output weights, or a combination of both
