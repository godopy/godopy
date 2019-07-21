from .headers.gdnative_api cimport *
from .globals cimport gdapi, nativescript_1_1_api, _python_language_index
from .core_cctypes cimport Vector2

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
    cdef void _set_position(Node2D self, Vector2 position):
        cdef PyObject *ptr = <PyObject *>self
        ___pygodot_icall_void_Vector2(__mb.__Node2D__mb_set_position, (<_Wrapped>ptr)._owner, position)

    def set_position(self, position):  # make namedtuple
        cdef float x, y
        x, y = position
        cdef Vector2 c_position = Vector2(x, y)
        cdef PyObject *ptr = <PyObject *>self
        ___pygodot_icall_void_Vector2(__mb.__Node2D__mb_set_position, (<_Wrapped>ptr)._owner, c_position)

cdef class Sprite(Node2D):
    pass

cdef struct __method_bindings:
    godot_method_bind *__Node2D__mb_set_position

cdef __method_bindings __mb

cdef public void __init_python_method_bindings():
    __mb.__Node2D__mb_set_position = gdapi.godot_method_bind_get_method("Node2D", "set_position")

cdef public void __register_python_types():
    cdef int i = _python_language_index
    cdef const godot_gdnative_ext_nativescript_1_1_api_struct *ns11api = nativescript_1_1_api

    print("Python language index", i)

    ns11api.godot_nativescript_set_global_type_tag(i, "Object", <const void *>Object)
    ns11api.godot_nativescript_set_global_type_tag(i, "Node", <const void *>Node)
    ns11api.godot_nativescript_set_global_type_tag(i, "CanvasItem", <const void *>CanvasItem)
    ns11api.godot_nativescript_set_global_type_tag(i, "Node2D", <const void *>Node2D)
    ns11api.godot_nativescript_set_global_type_tag(i, "Sprite", <const void *>Sprite)
