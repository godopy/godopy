from godot_headers.gdnative_api cimport *
from .globals cimport gdapi
from .cctypes cimport Vector2

from ._core cimport _Wrapped
from .__icalls cimport ___pygodot_icall_void_Vector2

from cpython.object cimport PyObject

cdef class Object(_Wrapped):
    def __cinit__(self):
        self.__CLASS_IS_SCRIPT = 0

cdef class Node(Object):
    pass

cdef class CanvasItem(Object):
    pass

cdef class Reference(Object):
    pass

cdef class Node2D(CanvasItem):
    cdef void set_position(Node2D self, Vector2 position):
        cdef PyObject *ptr = <PyObject *>self
        ___pygodot_icall_void_Vector2(__mb.__Node2D__mb_set_position, (<_Wrapped>ptr)._owner, position)


cdef class Sprite(Node2D):
    pass

cdef struct __method_bindings:
    godot_method_bind *__Node2D__mb_set_position

cdef __method_bindings __mb

cdef public void __init_python_method_bindings():
    __mb.__Node2D__mb_set_position = gdapi.godot_method_bind_get_method("Node2D", "set_position")
