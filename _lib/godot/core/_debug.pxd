from cpython.object cimport PyObject

cdef extern from "Python.h":
    Py_ssize_t Py_REFCNT(object ob)


cdef inline __ob_refcnt(object ob):
    return Py_REFCNT(ob)
