#include <Python.h>
#include "structmember.h"
#include "bilingual.hpp"
#include "numpy/arrayobject.h"

// MonoModel class

typedef struct {
    PyObject_HEAD
    MonolingualModel *model;
} MonoModel;

static void monolingual_dealloc(MonoModel *self) {
    delete self->model;
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject * monolingual_new(PyTypeObject *type, PyObject *args, PyObject *keywds) {
    MonoModel *self;
    self = (MonoModel *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int monolingual_init(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *model_file = NULL;
    self->model = new MonolingualModel();
    Config *config = &self->model->config;

    static char *kwlist[] = {"name", "learning_rate", "dimension", "min_count", "iterations", "window_size", "threads",
                             "subsampling", "verbose", "hierarchical_softmax", "skip_gram", "negative", "sent_vector",
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|sfiiiiifiiiii", kwlist, &model_file,
                                     &config->learning_rate, &config->dimension,
                                     &config->min_count, &config->iterations,
                                     &config->window_size, &config->threads,
                                     &config->subsampling, &config->verbose,
                                     &config->hierarchical_softmax, &config->skip_gram,
                                     &config->negative, &config->sent_vector))
        return NULL;


    if (model_file != NULL) {
        try {
            self->model = new MonolingualModel(model_file);  // this overwrites config

            // parse parameters again to overwrite the configuration of the loaded model
            config = &self->model->config;
            PyArg_ParseTupleAndKeywords(args, keywds, "|sfiiiiifiiiii", kwlist, &model_file,
                                        &config->learning_rate, &config->dimension,
                                        &config->min_count, &config->iterations,
                                        &config->window_size, &config->threads,
                                        &config->subsampling, &config->verbose,
                                        &config->hierarchical_softmax, &config->skip_gram,
                                        &config->negative, &config->sent_vector);
        } catch (...) {
            PyErr_SetString(PyExc_Exception, "Couldn't load model");
            return NULL;
        }
    }

    return 0;
}

static PyMemberDef monolingual_members[] = {
    {NULL}
};

static PyObject * monolingual_train(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;
    bool initialize = true;

    static char *kwlist[] = {"name", "initialize", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|b", kwlist, &filename, &initialize))
        return NULL;

    try {
        self->model->train(string(filename), initialize);
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't train model");
        return NULL;
    }
}

static PyObject * monolingual_load(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;

    static char *kwlist[] = {"name", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &filename))
        return NULL;

    try {
        self->model->load(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_IOError, "Couldn't load model");
        return NULL;
    }
}

static PyObject * monolingual_save(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;

    static char *kwlist[] = {"name", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &filename))
        return NULL;

    try {
        self->model->save(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_IOError, "Couldn't save model");
        return NULL;
    }
}

static PyObject * monolingual_save_vectors(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;
    int policy = 0;

    static char *kwlist[] = {"name", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|i", kwlist, &filename, &policy))
        return NULL;

    try {
        self->model->saveVectors(string(filename), policy);
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_IOError, "Couldn't save embeddings");
        return NULL;
    }
}

static PyObject * monolingual_save_vectors_bin(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;
    int policy = 0;

    static char *kwlist[] = {"name", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|i", kwlist, &filename, &policy))
        return NULL;

    try {
        self->model->saveVectorsBin(string(filename), policy);
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_IOError, "Couldn't save embeddings");
        return NULL;
    }
}

static PyObject * vec_to_numpy(vec arr) {
    float *a = (float *) malloc(sizeof(float) * arr.size());

    for (int i = 0; i < arr.size(); i++) {
        a[i] = arr[i];
    }

    npy_intp size = arr.size();
    PyArrayObject * nparr = (PyArrayObject *) PyArray_SimpleNewFromData(1, &size, NPY_FLOAT, a);
    nparr->flags |= NPY_OWNDATA;  // to avoid memory leaks
    //TODO: check memory leaks

    return PyArray_Return(nparr);
}

static vec numpy_to_vec(PyObject * arr) {
    int size = ((PyArrayObject *) arr)->dimensions[0]; // FIXME
    const char *data = ((PyArrayObject *) arr)->data;
    vector<float> v(data, data + size);
    return vec(v);
}

static PyObject * monolingual_sent_vec(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *sentence;

    static char *kwlist[] = {"sequence", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &sentence))
        return NULL;

    try {
        return vec_to_numpy(self->model->sentVec(string(sentence)));
    } catch(...) {
        PyErr_SetString(PyExc_ValueError, "Too short sentence / out of vocabulary words");
        return NULL;
    }
}

static PyObject * monolingual_word_vec(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *word;
    int policy = 0;

    static char *kwlist[] = {"word", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|i", kwlist, &word, &policy))
        return NULL;

    try {
        return vec_to_numpy(self->model->wordVec(string(word), policy));
    } catch(...) {
        PyErr_SetString(PyExc_ValueError, "Out of vocabulary word");
        return NULL;
    }
}

static PyObject * monolingual_similarity(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *word1;
    const char *word2;
    int policy = 0;

    static char *kwlist[] = {"word1", "word2", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &word1, &word2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->similarity(string(word1), string(word2), policy));
}

static PyObject * monolingual_distance(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *word1;
    const char *word2;
    int policy = 0;

    static char *kwlist[] = {"word1", "word2", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &word1, &word2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->distance(string(word1), string(word2), policy));
}

static PyObject * monolingual_similarity_sentence(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *seq1;
    const char *seq2;
    int policy = 0;

    static char *kwlist[] = {"sequence1", "sequence2", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &seq1, &seq2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->similaritySentence(string(seq1), string(seq2), policy));
}

static PyObject * monolingual_similarity_ngrams(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *seq1;
    const char *seq2;
    int policy = 0;

    static char *kwlist[] = {"sequence1", "sequence2", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &seq1, &seq2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->similarityNgrams(string(seq1), string(seq2), policy));
}

static PyObject * monolingual_closest(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *word;
    int n = 10;
    int policy = 0;

    static char *kwlist[] = {"word", "n", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|ii", kwlist, &word, &n, &policy))
        return NULL;

    vector<pair<string, float>> table = self->model->closest(string(word), n, policy);
    PyObject * res = PyList_New(0);
    for (auto it = table.begin(); it != table.end(); ++it) {
        PyObject * item = Py_BuildValue("sf", it->first.c_str(), it->second);
        PyList_Append(res, item);
    }
    return res;
}

static PyObject * monolingual_vocabulary(MonoModel *self) {
    PyObject * res = PyList_New(0);
    vector<pair<string, int>> words = self->model->getWords();
    for (auto it = words.begin(); it != words.end(); ++it) {
        PyObject * item = Py_BuildValue("s", it->first.c_str());
        PyList_Append(res, item);
    }
    return res;
}

static PyObject * monolingual_counts(MonoModel *self) {
    PyObject * res = PyList_New(0);
    vector<pair<string, int>> words = self->model->getWords();
    for (auto it = words.begin(); it != words.end(); ++it) {
        PyObject * item = Py_BuildValue("si", it->first.c_str(), it->second);
        PyList_Append(res, item);
    }
    return res;
}

static PyObject * monolingual_soft_WER(MonoModel *self, PyObject *args, PyObject *keywds) {
    const char *seq1;
    const char *seq2;
    int policy = 0;

    static char *kwlist[] = {"sequence1", "sequence2", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &seq1, &seq2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->softWER(string(seq1), string(seq2), policy));
}

static PyObject * monolingual_getdimension(MonoModel *self, void *closure) {
    PyObject * dim = Py_BuildValue("i", self->model->getDimension());
    return dim;
}

static PyObject * monolingual_set_config(MonoModel *self, PyObject *args, PyObject *keywds) {
    Config* config = &self->model->config;

    static char *kwlist[] = {"learning_rate", "dimension", "min_count", "iterations", "window_size", "threads",
                             "subsampling", "verbose", "hierarchical_softmax", "skip_gram", "negative", "sent_vector",
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|fiiiiifbbbib", kwlist,
                                     &config->learning_rate, &config->dimension,
                                     &config->min_count, &config->iterations,
                                     &config->window_size, &config->threads,
                                     &config->subsampling, &config->verbose,
                                     &config->hierarchical_softmax, &config->skip_gram,
                                     &config->negative, &config->sent_vector))
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * monolingual_get_config(MonoModel *self) {
    Config* config = &self->model->config;
    PyObject *res = PyDict_New();
    PyDict_SetItemString(res, "learning_rate", Py_BuildValue("f", config->learning_rate));
    PyDict_SetItemString(res, "dimension", Py_BuildValue("i", config->dimension));
    PyDict_SetItemString(res, "min_count", Py_BuildValue("i", config->min_count));
    PyDict_SetItemString(res, "iterations", Py_BuildValue("i", config->iterations));
    PyDict_SetItemString(res, "window_size", Py_BuildValue("i", config->window_size));
    PyDict_SetItemString(res, "threads", Py_BuildValue("i", config->threads));
    PyDict_SetItemString(res, "subsampling", Py_BuildValue("f", config->subsampling));
    PyDict_SetItemString(res, "verbose", config->verbose ? Py_True : Py_False);
    PyDict_SetItemString(res, "hierarchical_softmax", config->hierarchical_softmax ? Py_True : Py_False);
    PyDict_SetItemString(res, "skip_gram", config->skip_gram ? Py_True : Py_False);
    PyDict_SetItemString(res, "negative", Py_BuildValue("i", config->negative));
    PyDict_SetItemString(res, "sent_vector", config->sent_vector ? Py_True : Py_False);
    return res;
}

static PyMethodDef monolingual_methods[] = {
    {"train", (PyCFunction)monolingual_train, METH_VARARGS|METH_KEYWORDS,
     "train(name, initialize=True) -- train the model with given training file"
    },
    {"load", (PyCFunction)monolingual_load, METH_VARARGS|METH_KEYWORDS,
     "load(name) -- load entire model from disk"
    },
    {"save", (PyCFunction)monolingual_save, METH_VARARGS|METH_KEYWORDS,
     "save(name) -- save entire model to disk"
    },
    {"sent_vec", (PyCFunction)monolingual_sent_vec, METH_VARARGS|METH_KEYWORDS,
     "sent_vec(sequence) -- online paragraph vector for given whitespace delimited sequence"
    },
    {"word_vec", (PyCFunction)monolingual_word_vec, METH_VARARGS|METH_KEYWORDS,
     "word_vec(word, policy=0) -- get vector representation of word\n\n"
     "policy: 0) take the input weights\n"
     "        1) concatenation of input and output weights\n"
     "        2) sum of input and output weights\n"
     "        3) output weights\n"
    },
    {"save_vectors", (PyCFunction)monolingual_save_vectors, METH_VARARGS|METH_KEYWORDS,
     "save_vectors(name, policy=0) -- save word embeddings to disk in text format"
    },
    {"save_vectors_bin", (PyCFunction)monolingual_save_vectors_bin, METH_VARARGS|METH_KEYWORDS,
     "save_vectors_bin(name, policy=0) -- save word embeddings to disk in binary format"
    },
    {"closest", (PyCFunction)monolingual_closest, METH_VARARGS|METH_KEYWORDS,
     "closest(word, n=10, policy=0) -- get list of closest words to given word, according to cosine similarity"
    },
    {"similarity", (PyCFunction)monolingual_similarity, METH_VARARGS|METH_KEYWORDS,
     "similarity(word1, word2, policy=0) -- get cosine similarity between word1 and word2"
    },
    {"distance", (PyCFunction)monolingual_distance, METH_VARARGS|METH_KEYWORDS,
     "distance(word1, word2, policy=0) -- get distance between word1 and word2"
    },
    {"similarity_sentence", (PyCFunction)monolingual_similarity_sentence, METH_VARARGS|METH_KEYWORDS,
     "similarity_sentence(sequence1, sequence2, policy=0) -- get similarity between two sequences\n\n"
     "This method computes the cosine similarity between the bag-of-words representations of sentence1 and sentence2"
    },
    {"similarity_ngrams", (PyCFunction)monolingual_similarity_ngrams, METH_VARARGS|METH_KEYWORDS,
     "similarity_ngrams(sequence1, sequence2) -- get similarity between two sequences of equal size"
    },
    {"vocabulary", (PyCFunction)monolingual_vocabulary, METH_NOARGS,
     "vocabulary() -- get vocabulary list"
    },
    {"counts", (PyCFunction)monolingual_counts, METH_NOARGS,
     "counts() -- get list of words in vocabulary with their counts"
    },
    {"soft_WER", (PyCFunction)monolingual_soft_WER, METH_VARARGS|METH_KEYWORDS,
     "soft_WER(sequence1, sequence2) -- get soft word error rate between two sequences"
    },
    {"set_config", (PyCFunction)monolingual_set_config, METH_VARARGS|METH_KEYWORDS,
     "set_config(dimension=None, iterations=None, threads=None, ...) -- change model configuration"
    },
    {"get_config", (PyCFunction)monolingual_get_config, METH_NOARGS,
     "get_config() -- get configuration dict (read only)"
    },
    {NULL}
};

static PyGetSetDef monolingual_getseters[] = {
    {"dimension", (getter)monolingual_getdimension, NULL, "embedding dimension", NULL},
    {NULL}
};

static PyTypeObject MonoModelType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "MonolingualModel",         /*tp_name*/
    sizeof(MonoModel),            /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)monolingual_dealloc, /*tp_dealloc*/
    0,                            /*tp_print*/
    0,                            /*tp_getattr*/
    0,                            /*tp_setattr*/
    0,                            /*tp_compare*/
    0,                            /*tp_repr*/
    0,                            /*tp_as_number*/
    0,                            /*tp_as_sequence*/
    0,                            /*tp_as_mapping*/
    0,                            /*tp_hash */
    0,                            /*tp_call*/
    0,                            /*tp_str*/
    0,                            /*tp_getattro*/
    0,                            /*tp_setattro*/
    0,                            /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MonoModel objects",           /* tp_doc */
    0,		                      /* tp_traverse */
    0,		                      /* tp_clear */
    0,		                      /* tp_richcompare */
    0,		                      /* tp_weaklistoffset */
    0,		                      /* tp_iter */
    0,		                      /* tp_iternext */
    monolingual_methods,             /* tp_methods */
    monolingual_members,             /* tp_members */
    monolingual_getseters,          /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)monolingual_init,      /* tp_init */
    0,                            /* tp_alloc */
    monolingual_new,                 /* tp_new */
};

// BiModel class

typedef struct {
    PyObject_HEAD
    BilingualModel *model;
    PyObject *src_model;
    PyObject *trg_model;
} BiModel;

static void bilingual_dealloc(BiModel *self) {
    delete self->model;
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject * bilingual_new(PyTypeObject *type, PyObject *args, PyObject *keywds) {
    BiModel *self;
    self = (BiModel *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int bilingual_init(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *model_file = NULL;

    static char *kwlist[] = {"name", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist, &model_file))
        return NULL;

    try {
        if (model_file != NULL) {
            self->model = new BilingualModel(model_file);
        } else {
            self->model = new BilingualModel();
        }
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't load model");
        return NULL;
    }

    self->src_model = monolingual_new(&MonoModelType, NULL, NULL);
    ((MonoModel *)self->src_model)->model = &self->model->src_model;

    self->trg_model = monolingual_new(&MonoModelType, NULL, NULL);
    ((MonoModel *)self->trg_model)->model = &self->model->trg_model;

    return 0;
}

static PyObject * bilingual_train(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *src_file, *trg_file;
    bool initialize = true;

    static char *kwlist[] = {"src_name", "trg_name", "initialize", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|b", kwlist, &src_file, &trg_file, &initialize))
        return NULL;

    try {
        self->model->train(string(src_file), string(trg_file), initialize);
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't train model");
        return NULL;
    }
}

static PyObject * bilingual_save(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;

    static char *kwlist[] = {"name", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &filename))
        return NULL;

    try {
        self->model->save(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't save model");
        return NULL;
    }
}

static PyObject * bilingual_load(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *filename;

    static char *kwlist[] = {"name", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &filename))
        return NULL;

    try {
        self->model->load(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't load model");
        return NULL;
    }
}

static PyObject * bilingual_similarity(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *word1;
    const char *word2;
    int policy = 0;

    static char *kwlist[] = {"src_word", "trg_word", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &word1, &word2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->similarity(string(word1), string(word2), policy));
}

static PyObject * bilingual_distance(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *word1;
    const char *word2;
    int policy = 0;

    static char *kwlist[] = {"src_word", "trg_word", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &word1, &word2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->distance(string(word1), string(word2), policy));
}

static PyObject * bilingual_similarity_sentence(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *seq1;
    const char *seq2;
    int policy = 0;

    static char *kwlist[] = {"src_sequence", "trg_sequence", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &seq1, &seq2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->similaritySentence(string(seq1), string(seq2), policy));
}

static PyObject * bilingual_similarity_ngrams(BiModel *self, PyObject *args, PyObject *keywds) {
    const char *seq1;
    const char *seq2;
    int policy = 0;

    static char *kwlist[] = {"src_sequence", "trg_sequence", "policy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|i", kwlist, &seq1, &seq2, &policy))
        return NULL;

    return PyFloat_FromDouble(self->model->similarityNgrams(string(seq1), string(seq2), policy));
}

static PyMemberDef bilingual_members[] = {
    {"src_model", T_OBJECT_EX, offsetof(BiModel, src_model), 0, "Source model"},
    {"trg_model", T_OBJECT_EX, offsetof(BiModel, trg_model), 0, "Target model"},
};

static PyMethodDef bilingual_methods[] = {
    {"train", (PyCFunction)bilingual_train, METH_VARARGS|METH_KEYWORDS,
     "Train a bilingual model"
    },
    {"save", (PyCFunction)bilingual_save, METH_VARARGS|METH_KEYWORDS,
     "Save model to disk"
    },
    {"load", (PyCFunction)bilingual_load, METH_VARARGS|METH_KEYWORDS,
     "Load model from disk"
    },
    {"similarity", (PyCFunction)bilingual_similarity, METH_VARARGS|METH_KEYWORDS,
     "Similarity between two words"
    },
    {"distance", (PyCFunction)bilingual_distance, METH_VARARGS|METH_KEYWORDS,
     "Distance between two words"
    },
    {"similarity_sentence", (PyCFunction)bilingual_similarity_sentence, METH_VARARGS|METH_KEYWORDS,
     "Similarity between two sequences"
    },
    {"similarity_ngrams", (PyCFunction)bilingual_similarity_ngrams, METH_VARARGS|METH_KEYWORDS,
     "Similarity between two sequences of n-grams"
    },
    {NULL}
};

static PyTypeObject BiModelType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "BilingualModel",                /*tp_name*/
    sizeof(BiModel),                /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)bilingual_dealloc, /*tp_dealloc*/
    0,                            /*tp_print*/
    0,                            /*tp_getattr*/
    0,                            /*tp_setattr*/
    0,                            /*tp_compare*/
    0,                            /*tp_repr*/
    0,                            /*tp_as_number*/
    0,                            /*tp_as_sequence*/
    0,                            /*tp_as_mapping*/
    0,                            /*tp_hash */
    0,                            /*tp_call*/
    0,                            /*tp_str*/
    0,                            /*tp_getattro*/
    0,                            /*tp_setattro*/
    0,                            /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "BiModel objects",              /* tp_doc */
    0,		                      /* tp_traverse */
    0,		                      /* tp_clear */
    0,		                      /* tp_richcompare */
    0,		                      /* tp_weaklistoffset */
    0,		                      /* tp_iter */
    0,		                      /* tp_iternext */
    bilingual_methods,                /* tp_methods */
    bilingual_members,                /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)bilingual_init,         /* tp_init */
    0,                            /* tp_alloc */
    bilingual_new,                    /* tp_new */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initmultivec(void)
{
    PyObject* m;

    if (PyType_Ready(&MonoModelType) < 0 || PyType_Ready(&BiModelType) < 0)
        return;

    m = Py_InitModule3("multivec", monolingual_methods, "");
    import_array(); // for numpy

    Py_INCREF(&MonoModelType);
    Py_INCREF(&BiModelType);
    PyModule_AddObject(m, "MonolingualModel", (PyObject *)&MonoModelType);
    PyModule_AddObject(m, "BilingualModel", (PyObject *)&BiModelType);
}
