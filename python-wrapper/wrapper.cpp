#include <Python.h>
#include "structmember.h"
#include "bivec.h"
#include "numpy/arrayobject.h"

// Word2vec class

typedef struct {
    PyObject_HEAD
    MonolingualModel *model;
} Word2vec;

static void word2vec_dealloc(Word2vec *self) {
    delete self->model;
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject * word2vec_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Word2vec *self;
    self = (Word2vec *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int word2vec_init(Word2vec *self, PyObject *args, PyObject *kwds) {
    self->model = new MonolingualModel();
    return 0;
}

static PyMemberDef word2vec_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject * word2vec_train(Word2vec *self, PyObject *args) {
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

static PyObject * word2vec_load(Word2vec *self, PyObject *args) {
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

static PyObject * word2vec_save(Word2vec *self, PyObject *args) {
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

static PyObject * word2vec_save_embeddings(Word2vec *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename))
        return NULL;

    try {
        self->model->saveEmbeddings(string(filename));
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

static PyObject * word2vec_sent_vec(Word2vec *self, PyObject *args) {
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

static PyObject * word2vec_word_vec(Word2vec *self, PyObject *args) {
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

static PyMethodDef word2vec_methods[] = {
    {"train", (PyCFunction)word2vec_train, METH_VARARGS, "Train the model"},
    {"load", (PyCFunction)word2vec_load, METH_VARARGS, "Load model from the disk"},
    {"save", (PyCFunction)word2vec_save, METH_VARARGS, "Save model to disk"},
    {"sent_vec", (PyCFunction)word2vec_sent_vec, METH_VARARGS, "Inference step for paragraphs"},
    {"word_vec", (PyCFunction)word2vec_word_vec, METH_VARARGS, "Word embedding"},
    {"save_embeddings", (PyCFunction)word2vec_save_embeddings, METH_VARARGS, "Save word embeddings in a word2vec compliant format"},
    {NULL}  /* Sentinel */
};

static PyTypeObject Word2vecType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "word2vec.Word2vec",          /*tp_name*/
    sizeof(Word2vec),             /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)word2vec_dealloc, /*tp_dealloc*/
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
    "Word2vec objects",           /* tp_doc */
    0,		                      /* tp_traverse */
    0,		                      /* tp_clear */
    0,		                      /* tp_richcompare */
    0,		                      /* tp_weaklistoffset */
    0,		                      /* tp_iter */
    0,		                      /* tp_iternext */
    word2vec_methods,             /* tp_methods */
    word2vec_members,             /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)word2vec_init,      /* tp_init */
    0,                            /* tp_alloc */
    word2vec_new,                 /* tp_new */
};

// Bivec class

typedef struct {
    PyObject_HEAD
    BilingualModel *model;
    PyObject *src_model;
    PyObject *trg_model;
} Bivec;

static void bivec_dealloc(Bivec *self) {
    delete self->model;
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject * bivec_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Bivec *self;
    self = (Bivec *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int bivec_init(Bivec *self, PyObject *args, PyObject *kwds) {
    self->model = new BilingualModel();

    self->src_model = word2vec_new(&Word2vecType, NULL, NULL);
    ((Word2vec *)self->src_model)->model = &self->model->src_model;

    self->trg_model = word2vec_new(&Word2vecType, NULL, NULL);
    ((Word2vec *)self->trg_model)->model = &self->model->trg_model;

    return 0;
}

static PyObject * bivec_train(Bivec *self, PyObject *args) {
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

static PyObject * bivec_load(Bivec *self, PyObject *args) {
    const char *src_file, *trg_file;
    if (!PyArg_ParseTuple(args, "ss", &src_file, &trg_file))
        return NULL;

    try {
        self->model->load(string(src_file), string(trg_file));
        return Py_None;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "Couldn't load model");
        return NULL;
    }
}

static PyMemberDef bivec_members[] = {
    {"src_model", T_OBJECT_EX, offsetof(Bivec, src_model), 0, "Source model"},
    {"trg_model", T_OBJECT_EX, offsetof(Bivec, trg_model), 0, "Target model"},
};

static PyMethodDef bivec_methods[] = {
    {"train", (PyCFunction)bivec_train, METH_VARARGS, "Train a bilingual model"},
    {"load", (PyCFunction)bivec_load, METH_VARARGS, "Load full model from disk"},
    {NULL}  /* Sentinel */
};

static PyTypeObject BivecType = {
    PyObject_HEAD_INIT(NULL)
    0,                            /*ob_size*/
    "bivec.Bivec",                /*tp_name*/
    sizeof(Bivec),                /*tp_basicsize*/
    0,                            /*tp_itemsize*/
    (destructor)bivec_dealloc, /*tp_dealloc*/
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
    "Bivec objects",              /* tp_doc */
    0,		                      /* tp_traverse */
    0,		                      /* tp_clear */
    0,		                      /* tp_richcompare */
    0,		                      /* tp_weaklistoffset */
    0,		                      /* tp_iter */
    0,		                      /* tp_iternext */
    bivec_methods,                /* tp_methods */
    bivec_members,                /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)bivec_init,         /* tp_init */
    0,                            /* tp_alloc */
    bivec_new,                    /* tp_new */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initword2vec(void)
{
    PyObject* m;

    Word2vecType.tp_new = PyType_GenericNew;
    BivecType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Word2vecType) < 0 || PyType_Ready(&BivecType) < 0)
        return;

    m = Py_InitModule3("word2vec", word2vec_methods, "");
    import_array(); // for numpy

    Py_INCREF(&Word2vecType);
    Py_INCREF(&BivecType);
    PyModule_AddObject(m, "Word2vec", (PyObject *)&Word2vecType);
    PyModule_AddObject(m, "Bivec", (PyObject *)&BivecType);
}
