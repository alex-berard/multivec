import numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "vec.hpp":
    cdef cppclass Vec:
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

    cdef cppclass BilingualConfig:
        BilingualConfig()
        float beta
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


cdef extern from "monolingual.hpp":
    cdef cppclass MonolingualModel:
        MonolingualModel()
        MonolingualModel(Config* config)
        Vec wordVec(const string&, int) except +
        Vec sentVec(const string&) except +
        void train(const string&, bool) except +
        void load(const string&) except +
        Config* config


cdef extern from "bilingual.hpp":
    cdef cppclass BilingualModel:
        BilingualModel()
        BilingualModel(BilingualConfig* config)
        void train(const string&, const string&, bool) except +
        void load(const string&) except +
        MonolingualModel src_model
        MonolingualModel trg_model
        BilingualConfig* config


cdef class MonoModel:
    cdef Config config
    cdef MonolingualModel model
    def __cinit__(self, name=None, **kwargs):
        self.config = Config()
        self.model = MonolingualModel(&self.config)
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

    def train(self, name, initialize=True):
        self.model.train(name, initialize)

    def load(self, name):
        self.model.load(name)

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


cdef class BiModel:
    cdef BilingualConfig config
    cdef BilingualModel model
    def __cinit__(self, name=None, **kwargs):
        self.config = BilingualConfig()
        self.model = BilingualModel(&self.config)
        if name is not None:
            self.model.load(name)

    def train(self, src_name, trg_name, initialize=True):
        self.model.train(src_name, trg_name, initialize)
    
    def load(self, name):
        self.model.load(name)

    property src_model:
        def __get__(self):
            model = MonoModel()
            model.model = self.model.src_model
            model.config = self.config
            return model
    property trg_model:
        def __get__(self):
            model = MonoModel()
            model.model = self.model.trg_model
            model.config = self.config
            return model
