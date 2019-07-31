from godot_headers.gdnative_api cimport godot_object

include "core/defs.pxi"

cdef class _Wrapped:
    cdef godot_object *_owner
    cdef void *_type_tag
