# TODO: Generate automatically

from godot_headers.gdnative_api cimport godot_method_bind, godot_object
from .cpp.core_types cimport Vector2

cdef extern from "__py_icalls.hpp" nogil:
    void ___pygodot_icall_void_Vector2(godot_method_bind *mb, godot_object *o, const Vector2&)
