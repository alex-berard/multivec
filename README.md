# MultiVec
C++ implementation of *word2vec*, *bivec*, and *paragraph vector*.

## Features

### Monolingual model
* Most of word2vec's features [1, 6]
* Evaluation on the *analogical reasoning task* (multithreaded version of word2vec's *compute-accuracy*)
* Online paragraph vector [2]
* Save & load full model, including configuration and vocabulary
* Python wrapper

### Bilingual model
* Bivec-like training with a parallel corpus [3, 7]
* Uses two monolingual models, all the monolingual features are available
    for source & target model
    * Including monolingual paragraph vector [4]
* Save & load full model, or source/target model
* Python wrapper

## Dependencies
* GCC 4.4+
* CMake 2.6+
* Boost.Program_options
* Boost.Serialization

On Ubuntu/Debian:

    apt-get install libboost-program-options-dev libboost-serialization-dev

## TODO
* better software architecture for paragraph vector/online paragraph vector
* paragraph vector: DBOW model (similar to skip-gram)
* paragraph vector: option to concatenate, sum or average with word vectors.
* incremental training: possibility to train without erasing the model
* GIZA alignment for bilingual model
* write a small linear algebra module
* use custom serialization to remove dependency to boost
* use getopt instead of program-options

## References
1. [Distributed Representations of Words and Phrases and their Compositionality](http://arxiv.org/abs/1310.4546), Mikolov et al. (2013)
2. [Distributed Representations of Sentences and Documents](http://arxiv.org/abs/1405.4053), Le and Mikolov (2014)
3. [Bilingual Word Representations with Monolingual Quality in Mind](http://stanford.edu/~lmthang/bivec/), Luong et al. (2015)
4. [Learning Distributed Representations for Multilingual Text Sequences](http://www.aclweb.org/anthology/W15-1512), Pham et al. (2015)
5. [BilBOWA: Fast Bilingual Distributed Representations without Word Alignments](http://arxiv.org/abs/1410.2455), Gouws et al. (2014)
6. [Word2vec project](https://code.google.com/p/word2vec/)
7. [Bivec project](http://stanford.edu/~lmthang/bivec/)
8. [Gensim word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
9. [Gensim doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
