from godot.core cimport _Wrapped
from godot_cpp.core cimport Vector2

cdef class Object(_Wrapped):
    cdef int __CLASS_IS_SCRIPT

cdef class CanvasItem(Object):
    pass

cdef class Node2D(CanvasItem):
    cdef void set_position(Node2D self, Vector2 position)

cdef class Sprite(Node2D):
    pass
