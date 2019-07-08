# TODO: Generate automatically

from godot_headers.gdnative_api_struct__gen cimport *
from godot_cpp.core cimport _Wrapped, Vector2

cdef extern from "Object.hpp" namespace "godot" nogil:
    cdef cppclass Object(_Wrapped):
        @staticmethod
        const char *___get_class_name()

cdef extern from "CanvasItem.hpp" namespace "godot" nogil:
    cdef cppclass CanvasItem(Object):
        @staticmethod
        const char *___get_class_name()

cdef extern from "Node2D.hpp" namespace "godot" nogil:
    cdef cppclass Node2D(CanvasItem):
        @staticmethod
        const char *___get_class_name()

        Vector2 get_position()
        void set_position(const Vector2 position)

cdef extern from "Sprite.hpp" namespace "godot" nogil:
    cdef cppclass Sprite(Node2D):
        @staticmethod
        const char *___get_class_name()
