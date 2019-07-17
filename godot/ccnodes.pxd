# TODO: Generate automatically

from .headers.gdnative_api cimport *
from .core_types cimport _Wrapped, Variant, Vector2, Array, String

cdef extern from "Object.hpp" namespace "godot" nogil:
    cdef cppclass Object(_Wrapped):
        Variant emit_signal(const char *signal, ...) except+

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
