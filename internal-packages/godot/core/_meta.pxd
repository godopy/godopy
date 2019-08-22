cdef extern from "Python.h":
    ctypedef class __builtin__.type [object PyTypeObject]:
        cdef dict tp_dict

    void PyType_Modified(type type)
