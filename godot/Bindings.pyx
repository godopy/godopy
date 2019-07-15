from godot_headers.gdnative_api_struct__gen cimport *

from godot_cpp.Global cimport gdapi, nativescript_1_1_api, _RegisterState__python_language_index
from godot_cpp.CoreTypes cimport Vector2
from godot_cpp.Bindings cimport Object as _Object
from godot_cpp.__icalls cimport ___godot_icall_void_Vector2

from godot.Godot cimport _Wrapped

from cpython.object cimport PyObject

cdef class Object(_Wrapped):
    def __cinit__(self):
        self.__CLASS_IS_SCRIPT = 0

cdef class CanvasItem(Object):
    pass

cdef class Node2D(CanvasItem):
    cdef void set_position(Node2D self, Vector2 position):
        # TODO: Use cpdef with Pythonized args
        cdef PyObject *_this = <PyObject *>self
        ___godot_icall_void_Vector2(__mb.__Node2D__mb_set_position, <_Object *>_this, position)

cdef class Sprite(Node2D):
    pass

cdef struct __method_bindings:
    godot_method_bind *__Node2D__mb_set_position

cdef __method_bindings __mb

cdef public void __init_python_method_bindings():
    __mb.__Node2D__mb_set_position = gdapi.godot_method_bind_get_method("Node2D", "set_position")

cdef public void __register_python_types():
    cdef int i = _RegisterState__python_language_index
    cdef const godot_gdnative_ext_nativescript_1_1_api_struct *ns11api = nativescript_1_1_api

    print("Python language index", i)

    ns11api.godot_nativescript_set_global_type_tag(i, "Object", <const void *>Object)
    ns11api.godot_nativescript_set_global_type_tag(i, "CanvasItem", <const void *>CanvasItem)
    ns11api.godot_nativescript_set_global_type_tag(i, "Node2D", <const void *>Node2D)
    ns11api.godot_nativescript_set_global_type_tag(i, "Sprite", <const void *>Sprite)
