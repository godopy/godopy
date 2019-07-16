# TODO: Generate automatically

from godot_headers.gdnative_api_struct__gen cimport *
from godot_cpp.CoreTypes cimport Vector2
from godot.Godot cimport _Wrapped

cdef extern from "__py_icalls.hpp" nogil:
    void ___pygodot_icall_void_Vector2(godot_method_bind *mb, _Wrapped inst, const Vector2&)
