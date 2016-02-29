#include <Python.h>
#include "structmember.h"
#include "bilingual.hpp"
#include "numpy/arrayobject.h"

// MonoModel class

typedef struct {
    PyObject_HEAD
    MonolingualModel *model;
} MonoModel;

static void monomodel_dealloc(MonoModel *self) {
    delete self->model;
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject * monomodel_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    MonoModel *self;
    self = (MonoModel *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int monomodel_init(MonoModel *self, PyObject *args, PyObject *kwds) {
    self->model = new MonolingualModel();
    return 0;
}

static PyMemberDef monomodel_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject * monomodel_train(MonoModel *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->train(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't train model");
        return NULL;
    }
}

static PyObject * monomodel_load(MonoModel *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->load(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_IOError, "Couldn't load model");
        return NULL;
    }
}

static PyObject * monomodel_save(MonoModel *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->save(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_IOError, "Couldn't save model");
        return NULL;
    }
}

static PyObject * monomodel_save_embeddings(MonoModel *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->saveVectorsBin(string(filename));
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

static PyObject * monomodel_sent_vec(MonoModel *self, PyObject *args) {
    const char *sentence;
    if (!PyArg_ParseTuple(args, "s", &sentence))
        return NULL;

    try {
        return vec_to_numpy(self->model->sentVec(string(sentence)));
    } catch(...) {
        PyErr_SetString(PyExc_ValueError, "Too short sentence / out of vocabulary words");
        return NULL;
    }
}

static PyObject * monomodel_word_vec(MonoModel *self, PyObject *args) {
    const char *word;
    if (!PyArg_ParseTuple(args, "s", &word))
        return NULL;

    try {
        return vec_to_numpy(self->model->wordVec(string(word)));
    } catch(...) {
        PyErr_SetString(PyExc_ValueError, "Out of vocabulary word");
        return NULL;
    }
}

static PyObject * monomodel_similarity(MonoModel *self, PyObject *args) {
    const char *word1;
    const char *word2;
    if (!PyArg_ParseTuple(args, "ss", &word1, &word2))
        return NULL;

    return PyFloat_FromDouble(self->model->similarity(string(word1), string(word2)));
}

static PyObject * monomodel_closest(MonoModel *self, PyObject *args) {
    const char *word;
    int n;
    if (!PyArg_ParseTuple(args, "si", &word, &n))
        return NULL;

    vector<pair<string, float>> table = self->model->closest(string(word), n);
    PyObject * res = PyList_New(0);
    for (auto it = table.begin(); it != table.end(); ++it) {
        PyObject * item = Py_BuildValue("sf", it->first.c_str(), it->second);
        PyList_Append(res, item);
    }
    return res;
}

static PyObject * monomodel_vocabulary(MonoModel *self) {
    PyObject * res = PyList_New(0);
    vector<pair<string, int>> words = self->model->getWords();
    for (auto it = words.begin(); it != words.end(); ++it) {
        PyObject * item = Py_BuildValue("s", it->first.c_str());
        PyList_Append(res, item);
    }
    return res;
}

static PyObject * monomodel_counts(MonoModel *self) {
    PyObject * res = PyList_New(0);
    vector<pair<string, int>> words = self->model->getWords();
    for (auto it = words.begin(); it != words.end(); ++it) {
        PyObject * item = Py_BuildValue("si", it->first.c_str(), it->second);
        PyList_Append(res, item);
    }
    return res;
}

static PyObject * monomodel_closest_from_vec(MonoModel *self, PyObject *args) {
    PyObject * vec;
    if (!PyArg_ParseTuple(args, "o&", vec, PyArray_Converter)) // FIXME
        return NULL;
    /*
    PyObject * res = PyList_New(0);
    vector<pair<string, int>> words = self->model->getWords();
    for (auto it = words.begin(); it != words.end(); ++it) {
        PyObject * item = Py_BuildValue("si", it->first.c_str(), it->second);
        PyList_Append(res, item);
    }
    return res;
    */
    return 0;
}

static PyObject * monomodel_dimension(MonoModel *self) {
    PyObject * dim = Py_BuildValue("i", self->model->getDimension());
    return dim;
}

static PyObject * monomodel_soft_edit_distance(MonoModel *self, PyObject *args) {
    const char *seq1;
    const char *seq2;
    if (!PyArg_ParseTuple(args, "ss", &seq1, &seq2))
        return NULL;

    return PyFloat_FromDouble(self->model->softEditDistance(string(seq1), string(seq2)));
}

static PyMethodDef monomodel_methods[] = {
    {"train", (PyCFunction)monomodel_train, METH_VARARGS, "Train the model"},
    {"load", (PyCFunction)monomodel_load, METH_VARARGS, "Load model from the disk"},
    {"save", (PyCFunction)monomodel_save, METH_VARARGS, "Save model to disk"},
    {"sent_vec", (PyCFunction)monomodel_sent_vec, METH_VARARGS, "Inference step for paragraphs"},
    {"word_vec", (PyCFunction)monomodel_word_vec, METH_VARARGS, "Word embedding"},
    {"save_embeddings", (PyCFunction)monomodel_save_embeddings, METH_VARARGS, "Save word embeddings in the word2vec format"},
    {"closest", (PyCFunction)monomodel_closest, METH_VARARGS, "Closest words to given word"},
    {"closest_from_vec", (PyCFunction)monomodel_closest_from_vec, METH_VARARGS, "Closest words to given vector"},
    {"similarity", (PyCFunction)monomodel_similarity, METH_VARARGS, "Similarity between two words"},
    {"vocabulary", (PyCFunction)monomodel_vocabulary, METH_NOARGS, "Get words in vocabulary"},
    {"counts", (PyCFunction)monomodel_counts, METH_NOARGS, "Get words in vocabulary with their counts"},
    {"dimension", (PyCFunction)monomodel_dimension, METH_NOARGS, "Dimension of the embeddings"},
    {"soft_edit_distance", (PyCFunction)monomodel_soft_edit_distance, METH_VARARGS, "Soft Levenshtein distance between two sequences"},
    {NULL}  /* Sentinel */
};

static PyTypeObject MonoModelType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "monomodel.MonoModel",         /*tp_name*/
    sizeof(MonoModel),            /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)monomodel_dealloc, /*tp_dealloc*/
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
    monomodel_methods,             /* tp_methods */
    monomodel_members,             /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)monomodel_init,      /* tp_init */
    0,                            /* tp_alloc */
    monomodel_new,                 /* tp_new */
};

// BiModel class

typedef struct {
    PyObject_HEAD
    BilingualModel *model;
    PyObject *src_model;
    PyObject *trg_model;
} BiModel;

static void bimodel_dealloc(BiModel *self) {
    delete self->model;
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject * bimodel_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BiModel *self;
    self = (BiModel *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int bimodel_init(BiModel *self, PyObject *args, PyObject *kwds) {
    self->model = new BilingualModel();

    self->src_model = monomodel_new(&MonoModelType, NULL, NULL);
    ((MonoModel *)self->src_model)->model = &self->model->src_model;

    self->trg_model = monomodel_new(&MonoModelType, NULL, NULL);
    ((MonoModel *)self->trg_model)->model = &self->model->trg_model;

    return 0;
}

static PyObject * bimodel_train(BiModel *self, PyObject *args) {
    const char *src_file, *trg_file;
    if (!PyArg_ParseTuple(args, "ss", &src_file, &trg_file))
        return NULL;

    try {
        self->model->train(string(src_file), string(trg_file));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't train model");
        return NULL;
    }
}

static PyObject * bimodel_save(BiModel *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->save(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't save model");
        return NULL;
    }
}

static PyObject * bimodel_load(BiModel *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->load(string(filename));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't load model");
        return NULL;
    }
}

static PyMemberDef bimodel_members[] = {
    {"src_model", T_OBJECT_EX, offsetof(BiModel, src_model), 0, "Source model"},
    {"trg_model", T_OBJECT_EX, offsetof(BiModel, trg_model), 0, "Target model"},
};

static PyMethodDef bimodel_methods[] = {
    {"train", (PyCFunction)bimodel_train, METH_VARARGS, "Train a bilingual model"},
    {"save", (PyCFunction)bimodel_save, METH_VARARGS, "Save model to disk"},
    {"load", (PyCFunction)bimodel_load, METH_VARARGS, "Load model from disk"},
    {NULL}  /* Sentinel */
};

static PyTypeObject BiModelType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "bimodel.BiModel",                /*tp_name*/
    sizeof(BiModel),                /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)bimodel_dealloc, /*tp_dealloc*/
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
    bimodel_methods,                /* tp_methods */
    bimodel_members,                /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)bimodel_init,         /* tp_init */
    0,                            /* tp_alloc */
    bimodel_new,                    /* tp_new */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initmultivec(void)
{
    PyObject* m;

    MonoModelType.tp_new = PyType_GenericNew;
    BiModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MonoModelType) < 0 || PyType_Ready(&BiModelType) < 0)
        return;

    m = Py_InitModule3("multivec", monomodel_methods, "");
    import_array(); // for numpy

    Py_INCREF(&MonoModelType);
    Py_INCREF(&BiModelType);
    PyModule_AddObject(m, "MonoModel", (PyObject *)&MonoModelType);
    PyModule_AddObject(m, "BiModel", (PyObject *)&BiModelType);
}
