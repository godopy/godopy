from godot_headers.gdnative_api_struct__gen cimport *
from cpython.object cimport PyObject

cdef extern from "PythonGlobal.hpp" namespace "pygodot":
    ctypedef class _core._Wrapped [object __pygodot___Wrapped]:
        cdef godot_object *_owner
        cdef size_t _type_tag

cdef extern from "Python.h":
    PyObject* PyUnicode_FromWideChar(wchar_t *w, Py_ssize_t size)
