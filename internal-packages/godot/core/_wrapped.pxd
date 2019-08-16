from godot_headers.gdnative_api cimport godot_object

cdef class _Wrapped:
    cdef godot_object *_owner
    cdef bint ___CLASS_IS_SCRIPT
    cdef bint ___CLASS_IS_SINGLETON
    cdef int ___CLASS_BINDING_LEVEL

cdef class _PyWrapped(_Wrapped):
    pass
