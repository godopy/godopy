# TODO: Generate automatically

from godot_headers.gdnative_api_struct__gen cimport *
from .CoreTypes cimport Vector2
from .Bindings cimport Object

cdef extern from "__icalls.hpp" nogil:
    void ___godot_icall_void_Vector2(godot_method_bind *mb, const Object *inst, const Vector2&)
