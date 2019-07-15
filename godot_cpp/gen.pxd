# TODO: Generate automatically

from godot_headers.gdnative_api_struct__gen cimport *
from godot_cpp.core cimport _Wrapped, Variant, Vector2, Array, String

cdef extern from "Object.hpp" namespace "godot" nogil:
    cdef cppclass Object(_Wrapped):
        Variant emit_signal(const char *signal, ...) except+

cdef extern from "__icalls.hpp" nogil:
    void ___godot_icall_void_Vector2(godot_method_bind *mb, const Object *inst, const Vector2&)

cdef extern from "CanvasItem.hpp" namespace "godot" nogil:
    cdef cppclass CanvasItem(Object):
        pass

cdef extern from "Node2D.hpp" namespace "godot" nogil:
    cdef cppclass Node2D(CanvasItem):
        Vector2 get_position() except+
        void set_position(const Vector2 position) except+

cdef extern from "Sprite.hpp" namespace "godot" nogil:
    cdef cppclass Sprite(Node2D):
        pass
