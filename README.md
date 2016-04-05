# MultiVec
C++ implementation of *word2vec*, *bivec*, and *paragraph vector*.

## Features

### Monolingual model
* Most of word2vec's features [1, 6]
* Evaluation on the *analogical reasoning task* (multithreaded version of word2vec's *compute-accuracy*)
* Batch and online paragraph vector [2]
* Save & load full model, including configuration and vocabulary
* Python wrapper

### Bilingual model
* Bivec-like training with a parallel corpus [3, 7]
* Save & load full model
* Trains two monolingual models, which can be exported and used by MultiVec
* Python wrapper

## Dependencies
* GCC 4.4+
* CMake 2.6+
* Python and Numpy headers for the Python wrapper

## Installation

    git clone https://github.com/eske/multivec.git
    mkdir multivec/build
    cd multivec/build
    cmake ..
    make
    cd ..

The `bin` directory should now contain 4 binaries:
* `multivec-mono` which is used to generate monolingual models;
* `multivec-bi` to generate bilingual models;
* `word2vec` which is a modified version of word2vec that matches our user interface;
* `compute-accuracy` to evaluate word embeddings on the analogical reasoning task (multithreaded version of word2vec's compute-accuracy program).

### Python wrapper

    cd python-wrapper
    make

Use from Python:

    python2
    >>> from multivec import MonoModel, BiModel
    >>> model = BiModel()
    >>> model.load('models/europarl.en-de.bin')
    >>> model.word_vec('germany')
    ...

## Usage examples
First create two directories `data` and `models` at the root of the project, where you will put the text corpora and trained models.
The script `scripts/prepare.sh` can be used to pre-process a corpus (punctuation normalization, tokenization and lowercasing).

    mkdir data
    mkdir models
    wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz -P data
    tar xzf data/de-en.tgz -C data
    scripts/prepare.py data/europarl-v7.de-en.en en > data/europarl.en
    scripts/prepare.py data/europarl-v7.de-en.de de > data/europarl.de

To train a monolingual model using text corpus `data/europarl.en`:

    bin/multivec-mono --train data/europarl.en --save models/europarl.en.bin --threads 16

To train a bilingual model using parallel corpus `data/europarl.en`, `data/europarl.de`:

    bin/multivec-bi --train-src data/europarl.en --train-trg data/europarl.de --save models/europarl.en-de.bin --threads 16

To load a bilingual model and export it to source and target monolingual models:

    bin/multivec-bi --load models/europarl.en-de.bin --save-src models/europarl.en.bin --save-trg models/europarl.en.bin

To evaluate a trained English model on the analogical reasoning task, first export the model to the word2vec format, then use `compute-accuracy`:

    bin/multivec-mono --load models/europarl.en.bin --save-vectors-bin models/vectors.bin
    bin/compute-accuracy models/vectors.bin 0 < word2vec/questions-words.txt

## TODO
* better software architecture for paragraph vector/online paragraph vector
* paragraph vector: DBOW model (similar to skip-gram)
* paragraph vector: option to concatenate, sum or average with word vectors on projection layer.
* incremental training: possibility to train without erasing the model
* GIZA alignment for bilingual model
* bilingual paragraph vector training

## Acknowledgement

This toolkit is part of the project KEHATH (https://kehath.imag.fr/) funded by the French National Research Agency.

## References
1. [Distributed Representations of Words and Phrases and their Compositionality](http://arxiv.org/abs/1310.4546), Mikolov et al. (2013)
2. [Distributed Representations of Sentences and Documents](http://arxiv.org/abs/1405.4053), Le and Mikolov (2014)
3. [Bilingual Word Representations with Monolingual Quality in Mind](http://stanford.edu/~lmthang/bivec/), Luong et al. (2015)
4. [Learning Distributed Representations for Multilingual Text Sequences](http://www.aclweb.org/anthology/W15-1512), Pham et al. (2015)
5. [BilBOWA: Fast Bilingual Distributed Representations without Word Alignments](http://arxiv.org/abs/1410.2455), Gouws et al. (2014)
6. [Word2vec project](https://code.google.com/p/word2vec/)
7. [Bivec project](http://stanford.edu/~lmthang/bivec/)
