import numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef extern from "vec.hpp":
    cdef cppclass Vec:
        Vec()
        Vec(vector[float])
        const float* data()
        int size()


cdef extern from "utils.hpp":
    cdef cppclass Config:
        Config()
        float learning_rate
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
        MonolingualModelCpp()
        MonolingualModelCpp(Config* config)
        Vec wordVec(const string&, int) except +
        Vec sentVec(const string&) except +
        void train(const string&, bool) except +
        void load(const string&) except +
        void save(const string&) except +
        void saveVectors(const string&, int) except +
        void saveVectorsBin(const string&, int) except +
        void saveSentVectors(const string&) except +
        float similarity(const string&, const string&, int) except +
        float distance(const string&, const string&, int) except +
        float similarityNgrams(const string&, const string&, int) except +
        float similaritySentence(const string&, const string&, int) except +
        float softWER(const string&, const string&, int) except +
        vector[pair[string, float]] closest(const Vec&, int, int) except +
        vector[pair[string, float]] closest(const string&, const vector[string]& words, int) except +
        vector[pair[string, float]] closest(const string&, int, int) except +
        vector[pair[string, int]] getWords()
        Config* config


cdef extern from "bilingual.hpp":
    cdef cppclass BilingualModelCpp "BilingualModel":
        BilingualModelCpp()
        BilingualModelCpp(BilingualConfig* config)
        void train(const string&, const string&, bool) except +
        void load(const string&) except +
        MonolingualModelCpp src_model
        MonolingualModelCpp trg_model
        BilingualConfig* config


cdef class MonolingualModel:
    cdef Config* config
    cdef MonolingualModelCpp model
    def __cinit__(self, name=None, **kwargs):
        self.config = new Config()
        self.model = MonolingualModelCpp(self.config)
        if name is not None:
            self.model.load(name)

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def word_vec(self, word, policy=0):
        cdef Vec vec = self.model.wordVec(word, policy)
        cdef float* data = vec.data()
        return np.array([data[i] for i in range(vec.size())])

    def sent_vec(self, sequence):
        cdef Vec vec = self.model.sentVec(sequence)
        cdef float* data = vec.data()
        return np.array([data[i] for i in range(vec.size())])

    def train(self, name, initialize=True): self.model.train(name, initialize)
    def load(self, name): self.model.load(name)
    def save(self, name): self.model.save(name)
    def save_vectors(self, name, policy=0): self.model.saveVectorsBin(name, policy)
    def save_vectors_bin(self, name, policy=0): self.model.saveVectorsBin(name, policy)
    def save_sent_vectors(self, name): self.model.saveSentVectors(name)
    
    def similarity(self, word1, word2, policy=0):
        return self.model.similarity(word1, word2, policy)
    def distance(self, word1, word2, policy=0):
        return self.model.distance(word1, word2, policy)
            
    def similarity_n_grams(self, seq1, seq2, policy=0):
        return self.model.similarityNgrams(seq1, seq2, policy)
    def similarity_bag_of_words(self, seq1, seq2, policy=0):
        return self.model.similaritySentence(seq1, seq2, policy)
    def soft_word_error_rate(self, seq1, seq2, policy=0):
        return self.model.softWER(seq1, seq2, policy)
    
    def closest(self, word, n=10, policy=0):
        cdef vector[pair[string, float]] res = self.model.closest(<const string&> word, <int> n, <int> policy)
        return list(res)
    def closest_to_vec(self, vec, n=10, policy=0):
        cdef Vec vec_cpp = Vec(<vector[float]> vec)
        res = self.model.closest(<const Vec&> vec_cpp, <int> n, <int> policy)
        return list(res)
    def closest_words(self, word, words, policy=0):
        cdef vector[pair[string, float]] res = self.model.closest(<const string&> word,
                                                                  <const vector[string]&> words,
                                                                  <int> policy)
        return list(res)
    def get_vocabulary(self):
        cdef vector[pair[string, int]] word_counts = self.model.getWords()
        return [w for w, _ in word_counts]
    def get_counts(self):
        cdef vector[pair[string, int]] word_counts = self.model.getWords()
        return dict(word_counts)
        
    property learning_rate:
        def __get__(self): return self.config.learning_rate
        def __set__(self, learning_rate): self.config.learning_rate = learning_rate
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
    cdef BilingualConfig* config
    cdef BilingualModelCpp model
    def __cinit__(self, name=None, **kwargs):
        self.config = new BilingualConfig()  # memory leak, or not?
        self.model = BilingualModelCpp(self.config)
        if name is not None:
            self.model.load(name)

    def train(self, src_name, trg_name, initialize=True):
        self.model.train(src_name, trg_name, initialize)
    
    def load(self, name):
        self.model.load(name)

    property src_model:
        def __get__(self):
            # create the model on the fly, because the reference can (in theory) change
            # during the lifetime of the object
            model = MonolingualModel()
            model.model = self.model.src_model
            model.config = self.config
            return model
    property trg_model:
        def __get__(self):
            model = MonolingualModel()
            model.model = self.model.trg_model
            model.config = self.config
            return model
