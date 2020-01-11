from cpython.object cimport PyTypeObject

cdef extern from "Python.h":
    void PyType_Modified(type type)

cdef inline dict __tp_dict(type type):
    # WARNING: This dict should be used only during type creation followed by PyType_Modified call
    # It is not safe to modify in any way after type initialization
    return <dict>(<PyTypeObject *>type).tp_dict
