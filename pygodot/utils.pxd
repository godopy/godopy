from godot_headers.gdnative_api cimport godot_string

cdef str godot_project_dir()
cdef bytes godot_string_to_bytes(const godot_string *s)

cdef int _init_dynamic_loading() except -1
