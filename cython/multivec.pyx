import numpy as np
from collections import OrderedDict
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool


cdef extern from "vec.hpp":
    cdef cppclass Vec:
        Vec()
        Vec(vector[float])
        const float* data()
        int size()


cdef extern from "utils.hpp":
    cdef cppclass Config:
        Config()
        float alpha
        int dimension
        int min_count
        int iterations
        int window_size
        int threads
        float subsampling
        int verbose
        int hierarchical_softmax
        int skip_gram
        int negative
        int sent_vector

    cdef cppclass BilingualConfig(Config):
        BilingualConfig()
        float beta


cdef extern from "monolingual.hpp":
    cdef cppclass MonolingualModelCpp "MonolingualModel":
        MonolingualModelCpp(Config*) except +
        Vec word_vec(const string&, int) except +
        Vec sent_vec(const string&) except +
        void train(const string&, bool) except +
        void load(const string&) except +
        void save(const string&) except +
        void save_vectors(const string&, int) except +
        void save_vectors_bin(const string&, int) except +
        void save_sent_vectors(const string&) except +
        float similarity(const string&, const string&, int) except +
        float distance(const string&, const string&, int) except +
        float similarity_ngrams(const string&, const string&, int) except +
        float similarity_bag_of_words(const string&, const string&, int) except +
        float similarity_syntax(const string&, const string&, const string&, const string&,
                                       const vector[float]&, const vector[float]&, float, int) except +
        float soft_word_error_rate(const string&, const string&, int) except +
        vector[pair[string, float]] closest(const Vec&, int, int) except +
        vector[pair[string, float]] closest(const string&, const vector[string]&, int) except +
        vector[pair[string, float]] closest(const string&, int, int) except +
        vector[pair[string, int]] get_word_counts() except +
        Config* config


cdef extern from "bilingual.hpp":
    cdef cppclass BilingualModelCpp "BilingualModel":
        BilingualModelCpp(BilingualConfig*) except +
        void train(const string&, const string&, const string&, bool) except +
        void load(const string&) except +
        void save(const string&) except +
        float similarity(const string&, const string&, int) except +
        float distance(const string&, const string&, int) except +
        float similarity_ngrams(const string&, const string&, int) except +
        float similarity_bag_of_words(const string&, const string&, int) except +
        float similarity_syntax(const string&, const string&, const string&, const string&,
                                const vector[float]&, const vector[float]&, float, int) except +
        vector[pair[string, float]] trg_closest(const string&, int, int) except +
        vector[pair[string, float]] src_closest(const string&, int, int) except +
        MonolingualModelCpp src_model
        MonolingualModelCpp trg_model
        BilingualConfig* config


cdef class MonolingualModel:
    """
    MonolingualModel(name=None, **kwargs)
    
    Parameters
    ----------
    name : path to an existing model. This model and its parameters
        (including vocabulary and configuration) will be loaded.
    kwargs : overwrite configuration of the model (see attributes)
    
    Attributes
    ----------
    alpha : initial learning rate, which will decay to zero during training (default: 0.05)
    dimension : dimension of the embeddings (default: 100)
    min_count : minimum count of a word in the training file to be put in the vocabulary (default: 5)
    iterations : number of training iterations (default: 5)
    window_size : size of the context window of CBOW and Skip-Gram (default: 5)
    threads : number of threads used in training (default: 4)
    subsampling : sub-sampling rate (default: 1e-03)
    verbose : display useful information during training (default: False)
    hierarchical_softmax : toggle the hierarchical softmax training objective (default: False)
    skip_gram : use Skip-Gram model instead of CBOW (default: False)
    negative : number of negative samples per positive sample in negative sampling (set to
        0 to disable negative sampling) (default: 5)
    sent_vector : include sentence vectors in training. This is an implementation of
        batch paragraph vector (default: False)
    
    Examples
    --------
    >>> model = MonolingualModel('models/news-commentary.en.bin', iterations=10)
    >>> model.iterations  # overwritten value
    10
    >>> model.dimension   # default value is 100, but loaded model has a different config
    40
    """
    cdef Config* config
    cdef MonolingualModelCpp* model
    cdef int alloc
    
    def __cinit__(self, name=None, **kwargs):
        # if alloc is False, manual deallocation of config and model
        self.alloc = True
        self.config = new Config()
        self.model = new MonolingualModelCpp(self.config)
    
    cdef set_members(self, MonolingualModelCpp* model, Config* config):
        # used by BilingualModel to initialize self.model and self.config to existing values
        del self.config
        del self.model
        self.config = config
        self.model = model
        self.alloc = False
        return self
    
    def __init__(self, name=None, **kwargs):
        if name is not None:
            self.model.load(name.encode('utf-8'))
        
        # overwrites previous configuration
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
        
    def __dealloc__(self):
        if self.alloc:  # we don't want to dealloc parent bilingual model's parameters
            del self.model
            del self.config

    def word_vec(self, word, policy=0):
        """
        word_vec(word, policy=0)
        
        Return the vector representation of word
        Raise RuntimeError if word is out of vocabulary
        
        Policy determines which weights to use:
            0) take the input weights
            1) concatenation of input and output weights
            2) sum of input and output weights
            3) output weights
        """
        cdef Vec vec = self.model.word_vec(word.encode('utf-8'), policy)
        cdef float* data = vec.data()
        return np.array([data[i] for i in range(vec.size())])

    def sent_vec(self, sequence):
        """
        sent_vec(sequence)
        
        Perform paragraph vector inference step on given sequence.
        Sequence must be a whitespace-delimited string of words.
        
        Raise RuntimeError if sequence is empty or all words are OOV.
        """
        cdef Vec vec = self.model.sent_vec(sequence.encode('utf-8'))
        cdef float* data = vec.data()
        return np.array([data[i] for i in range(vec.size())])

    def train(self, name, initialize=True):
        """
        train(name, initialize=True)
        
        Train model with training file of path `name`. This file must be word-tokenized,
        with one sentence per line.
        Raise RuntimeError if file is empty or cannot be opened.

        Initialize will create a new vocabulary from training file, and initialize the model's
        weight to random values.
        Set this value to False to continue training of an existing model (learning rate will
        be reset to its initial value, i.e. self.alpha)
        """
        self.model.train(name.encode('utf-8'), initialize)
        
    def load(self, name):
        """
        load(name)

        Load model from disk (path `name`). This model must have been saved with `MonolingualModel.save`.
        This function cannot load files in the word2vec format.

        The entire model, including configuration and vocabulary is loaded, and the existing
        parameters are overwritten.
        """
        self.model.load(name.encode('utf-8'))

    def save(self, name):
        """
        save(name)

        Save entire model to disk (path `name`). This saves the entire model, including configuration and 
        vocabulary into a binary format, specific to MultiVec. Those models can then be loaded
        from disk using `MonolingualModel.load`.
        """
        self.model.save(name.encode('utf-8'))

    def save_vectors(self, name, policy=0):
        """
        save_vectors(name)
        
        Save the word vectors to disk (path `name`) in the word2vec text format.
        
        The `policy` parameters
        decides which weights are used (see `MonolingualModel.word_vec` for details.)
        """
        self.model.save_vectors(name.encode('utf-8'), policy)

    def save_vectors_bin(self, name, policy=0):
        """
        save_vectors(name)
        
        Save the word vectors to disk (path `name`) in the word2vec binary format.
        
        The `policy` parameters
        decides which weights are used (see `MonolingualModel.word_vec` for details.)
        """
        self.model.save_vectors_bin(name.encode('utf-8'), policy)

    def save_sent_vectors(self, name):
        self.model.save_sent_vectors(name.encode('utf-8'))
    
    def similarity(self, word1, word2, policy=0):
        return self.model.similarity(word1.encode('utf-8'), word2.encode('utf-8'), policy)
    def distance(self, word1, word2, policy=0):
        return self.model.distance(word1.encode('utf-8'), word2.encode('utf-8'), policy)
            
    def similarity_ngrams(self, seq1, seq2, policy=0):
        return self.model.similarity_ngrams(seq1.encode('utf-8'), seq2.encode('utf-8'), policy)
    def similarity_bag_of_words(self, seq1, seq2, policy=0):
        return self.model.similarity_bag_of_words(seq1.encode('utf-8'), seq2.encode('utf-8'), policy)
    def similarity_syntax(self, seq1, seq2, tags1, tags2, idf1, idf2, alpha=0.0, policy=0):
        return self.model.similarity_syntax(seq1.encode('utf-8'), seq2.encode('utf-8'),
                                            tags1.encode('utf-8'), tags2.encode('utf-8'),
                                            idf1, idf2, alpha, policy)
    def soft_word_error_rate(self, seq1, seq2, policy=0):
        return self.model.soft_word_error_rate(seq1.encode('utf-8'), seq2.encode('utf-8'), policy)
    
    def closest(self, word, n=10, policy=0):
        cdef vector[pair[string, float]] res = self.model.closest(<const string&> word.encode('utf-8'), <int> n, <int> policy)
        return [(w.decode('utf-8'), c) for w, c in res]
    def closest_to_vec(self, vec, n=10, policy=0):
        cdef Vec vec_cpp = Vec(<vector[float]> vec)
        res = self.model.closest(<const Vec&> vec_cpp, <int> n, <int> policy)
        return [(w.decode('utf-8'), c) for w, c in res]
    def closest_words(self, word, words, policy=0):
        cdef vector[pair[string, float]] res = self.model.closest(<const string&> word.encode('utf-8'),
                                                                  <const vector[string]&> words,
                                                                  <int> policy)
        return [(w.decode('utf-8'), c) for w, c in res]
    def get_vocabulary(self):
        cdef vector[pair[string, int]] word_counts = self.model.get_word_counts()
        return [w.decode('utf-8') for w, _ in word_counts]
    def get_word_counts(self):
        cdef vector[pair[string, int]] word_counts = self.model.get_word_counts()
        return OrderedDict((w.decode('utf-8'), c) for w, c in word_counts)
        
    property alpha:
        def __get__(self): return self.config.alpha
        def __set__(self, alpha): self.config.alpha = alpha
    property dimension:
        def __get__(self): return self.config.dimension
        def __set__(self, dimension): self.config.dimension = dimension
    property min_count:
        def __get__(self): return self.config.min_count
        def __set__(self, min_count): self.config.min_count = min_count
    property iterations:
        def __get__(self): return self.config.iterations
        def __set__(self, iterations): self.config.iterations = iterations
    property window_size:
        def __get__(self): return self.config.window_size
        def __set__(self, window_size): self.config.window_size = window_size
    property threads:
        def __get__(self): return self.config.threads
        def __set__(self, threads): self.config.threads = threads
    property subsampling:
        def __get__(self): return self.config.subsampling
        def __set__(self, subsampling): self.config.subsampling = subsampling
    property verbose:
        def __get__(self): return self.config.verbose
        def __set__(self, verbose): self.config.verbose = verbose
    property hierarchical_softmax:
        def __get__(self): return self.config.hierarchical_softmax
        def __set__(self, hierarchical_softmax): self.config.hierarchical_softmax = hierarchical_softmax
    property skip_gram:
        def __get__(self): return self.config.skip_gram
        def __set__(self, skip_gram): self.config.skip_gram = skip_gram
    property negative:
        def __get__(self): return self.config.negative
        def __set__(self, negative): self.config.negative = negative
    property sent_vector:
        def __get__(self): return self.config.sent_vector
        def __set__(self, sent_vector): self.config.sent_vector = sent_vector


cdef class BilingualModel:
    """
    BilingualModel(name=None, **kwargs)
    
    Parameters
    ----------
    name : path to an existing model. This model and its parameters
        (including vocabulary and configuration) will be loaded.
    kwargs : overwrite configuration of the model (see attributes)
    
    Attributes
    ----------
    src_model : source monolingual model
    trg_model : target monolingual model
    beta : weight given to bilingual updates (compared to monolingual updates) (default: 1)
    alpha : initial learning rate, which will decay to zero during training (default: 0.05)
    dimension : dimension of the embeddings (default: 100)
    min_count : minimum count of a word in the training file to be put in the vocabulary (default: 5)
    iterations : number of training iterations (default: 5)
    window_size : size of the context window of CBOW and Skip-Gram (default: 5)
    threads : number of threads used in training (default: 4)
    subsampling : sub-sampling rate (default: 1e-03)
    verbose : display useful information during training (default: False)
    hierarchical_softmax : toggle hierarchical softmax training objective (default: False)
    skip_gram : use Skip-Gram model instead of CBOW (default: False)
    negative : number of negative samples per positive sample in negative sampling (set to
        0 to disable negative sampling) (default: 5)
    sent_vector : include sentence vectors in training. This is an implementation of
        batch paragraph vector (default: False)
    
    Examples
    --------
    >>> model = MonolingualModel('models/news-commentary.fr-en.bin', alpha=0.1)
    >>> model.alpha
    0.1
    >>> model.src_model
    <multivec.MonolingualModel at 0x7f515c0687b0>
    >>> model.src_model.alpha
    0.1
    """
    cdef BilingualConfig* config
    cdef BilingualModelCpp* model
    def __cinit__(self, name=None, **kwargs):
        self.config = new BilingualConfig()
        self.model = new BilingualModelCpp(self.config)
        if name is not None:
            self.model.load(name.encode('utf-8'))
    
        # overwrites previous configuration
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def __dealloc__(self):
        # careful: this will break children monolingual models (because they use the same config and c++ models)
        del self.model
        del self.config

    def train(self, src_name, trg_name, align_name=None, initialize=True):
        align_name = align_name or ''
        self.model.train(src_name.encode('utf-8'), trg_name.encode('utf-8'), align_name.encode('utf-8'), initialize)
    
    def save(self, name):
        self.model.save(name.encode('utf-8'))
    
    def load(self, name):
        self.model.load(name.encode('utf-8'))

    def similarity(self, src_word, trg_word, policy=0):
        return self.model.similarity(src_word.encode('utf-8'), trg_word.encode('utf-8'), policy)
    def distance(self, src_word, trg_word, policy=0):
        return self.model.similarity(src_word.encode('utf-8'), trg_word.encode('utf-8'), policy)
            
    def similarity_ngrams(self, src_seq, trg_seq, policy=0):
        return self.model.similarity_ngrams(src_seq.encode('utf-8'), trg_seq.encode('utf-8'), policy)
    def similarity_bag_of_words(self, src_seq, trg_seq, policy=0):
        return self.model.similarity_bag_of_words(src_seq.encode('utf-8'), trg_seq.encode('utf-8'), policy)
    def similarity_syntax(self, src_seq, trg_seq, src_tags, trg_tags, src_idf, trg_idf, alpha=0.0, policy=0):
        return self.model.similarity_syntax(src_seq.encode('utf-8'), trg_seq.encode('utf-8'),
                                            src_tags.encode('utf-8'), trg_tags.encode('utf-8'),
                                            src_idf, trg_idf, alpha, policy)
    def trg_closest(self, src_word, n=10, policy=0):
        cdef vector[pair[string, float]] res = self.model.trg_closest(<const string&> src_word.encode('utf-8'), <int> n, <int> policy)
        return [(w.decode('utf-8'), c) for w, c in res]
    def src_closest(self, trg_word, n=10, policy=0):
        cdef vector[pair[string, float]] res = self.model.src_closest(<const string&> trg_word.encode('utf-8'), <int> n, <int> policy)
        return [(w.decode('utf-8'), c) for w, c in res]

    property src_model:
        def __get__(self):
            # create the model on the fly, because the reference can (in theory) change
            # during the lifetime of the object
            return MonolingualModel().set_members(&self.model.src_model, self.config)
    property trg_model:
        def __get__(self):
            return MonolingualModel().set_members(&self.model.trg_model, self.config)
 
    property beta:
        def __get__(self): return self.config.beta           
        def __set__(self, beta): self.config.beta = beta
    property alpha:
        def __get__(self): return self.config.alpha
        def __set__(self, alpha): self.config.alpha = alpha
    property dimension:
        def __get__(self): return self.config.dimension
        def __set__(self, dimension): self.config.dimension = dimension
    property min_count:
        def __get__(self): return self.config.min_count
        def __set__(self, min_count): self.config.min_count = min_count
    property iterations:
        def __get__(self): return self.config.iterations
        def __set__(self, iterations): self.config.iterations = iterations
    property window_size:
        def __get__(self): return self.config.window_size
        def __set__(self, window_size): self.config.window_size = window_size
    property threads:
        def __get__(self): return self.config.threads
        def __set__(self, threads): self.config.threads = threads
    property subsampling:
        def __get__(self): return self.config.subsampling
        def __set__(self, subsampling): self.config.subsampling = subsampling
    property verbose:
        def __get__(self): return self.config.verbose
        def __set__(self, verbose): self.config.verbose = verbose
    property hierarchical_softmax:
        def __get__(self): return self.config.hierarchical_softmax
        def __set__(self, hierarchical_softmax): self.config.hierarchical_softmax = hierarchical_softmax
    property skip_gram:
        def __get__(self): return self.config.skip_gram
        def __set__(self, skip_gram): self.config.skip_gram = skip_gram
    property negative:
        def __get__(self): return self.config.negative
        def __set__(self, negative): self.config.negative = negative
    property sent_vector:
        def __get__(self): return self.config.sent_vector
        def __set__(self, sent_vector): self.config.sent_vector = sent_vector
