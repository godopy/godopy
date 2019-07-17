from .headers.gdnative_api cimport godot_object

cdef extern from "PythonGlobal.hpp" namespace "pygodot":
    ctypedef class pygodot._Wrapped [object __pygodot___Wrapped]:
        cdef godot_object *_owner
        cdef size_t _type_tag
