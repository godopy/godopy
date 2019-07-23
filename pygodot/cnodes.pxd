from .cctypes cimport Vector2
from ._core cimport _Wrapped

cdef class Object(_Wrapped):
    cdef int __CLASS_IS_SCRIPT

cdef class Node(Object):
    pass

cdef class CanvasItem(Object):
    pass

cdef class Reference(Object):
    pass

cdef class Node2D(CanvasItem):
    cdef void set_position(Node2D self, Vector2 position)

cdef class Sprite(Node2D):
    pass
