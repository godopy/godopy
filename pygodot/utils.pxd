from .headers.gdnative_api cimport godot_string

cdef bytes godot_string_to_bytes(const godot_string *s)
