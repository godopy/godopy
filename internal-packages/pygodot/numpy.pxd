# These #defines must be declared *before* #include <numpy/array_object.h>
cdef extern from *:
    '''\
#ifndef NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL PYGODOT_ARRAY_API
#endif
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API 0x0000000D
#endif
    '''

from cpython cimport object as cpython
from cpython cimport ref

cdef extern from "numpy/arrayobject.h":
    cdef cpython.PyTypeObject PyArray_Type

from numpy cimport *


cdef inline object array_new(type cls, int nd, npy_intp *dims, int typenum, npy_intp *strides, void *data, int itemsize, int flags):
    return PyArray_New(cls, nd, dims, typenum, strides, data, itemsize, flags, <object>NULL)


cdef inline object array_new_simple(int nd, npy_intp *dims, int typenum, void *data):
    return PyArray_New(<type>&PyArray_Type, nd, dims, typenum, NULL, data, 0, NPY_ARRAY_CARRAY, <object>NULL)


cdef inline object array_new_simple_readonly(int nd, npy_intp *dims, int typenum, void *data):
    return PyArray_New(<type>&PyArray_Type, nd, dims, typenum, NULL, data, 0, NPY_ARRAY_CARRAY_RO, <object>NULL)


cdef inline void set_array_base(ndarray arr, object base):
    ref.Py_INCREF(base) # important to do this before stealing the reference below!
    PyArray_SetBaseObject(arr, base)


cdef inline object get_array_base(ndarray arr):
    base = PyArray_BASE(arr)
    if base is NULL:
        return None
    return <object>base
