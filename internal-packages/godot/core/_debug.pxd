from cpython.object cimport PyObject

cdef inline __ob_refcnt(object obj):
    return (<PyObject *>obj).ob_refcnt
