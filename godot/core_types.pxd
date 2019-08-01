from godot_headers.gdnative_api cimport godot_object

include "core/defs.pxi"

cdef class _Wrapped:
    cdef godot_object *_owner

cdef class _PyWrapped:
    cdef godot_object *_owner

cdef dict CythonTagDB
cdef dict PythonTagDB

cdef register_cython_type(type cls)
cdef register_python_type(type cls)
cdef register_global_cython_type(type cls, str api_name)
cdef register_global_python_type(type cls, str api_name)
