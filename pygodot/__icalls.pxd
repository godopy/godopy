# TODO: Generate automatically

from .headers.gdnative_api cimport godot_method_bind, godot_object
from .core_cctypes cimport Vector2

cdef extern from "__py_icalls.hpp" nogil:
    void ___pygodot_icall_void_Vector2(godot_method_bind *mb, godot_object *o, const Vector2&)
