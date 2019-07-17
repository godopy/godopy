# TODO: Generate automatically

from .headers.gdnative_api cimport *
from .core_cctypes cimport Vector2
from ._core cimport _Wrapped

cdef extern from "__py_icalls.hpp" nogil:
    void ___pygodot_icall_void_Vector2(godot_method_bind *mb, _Wrapped inst, const Vector2&)
