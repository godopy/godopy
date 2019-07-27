from godot_headers.gdnative_api cimport godot_gdnative_ext_nativescript_1_1_api_struct
from .globals cimport gdapi, nativescript_1_1_api, _python_language_index
from . cimport cnodes
from .cpp.core_types cimport Vector2


class Object(cnodes.Object):
    pass

class Node(cnodes.Node, Object):
    pass

class Reference(cnodes.Reference, Node):
    pass

class CanvasItem(cnodes.CanvasItem, Object):
    pass

class Node2D(cnodes.Node2D, CanvasItem):
    def set_position(self, position):
        cdef float x, y
        x, y = position
        cnodes.Node2D.set_position(self, Vector2(x, y))

class Sprite(cnodes.Sprite, Node2D):
    pass

cdef public void __register_python_types():
    cdef int i = _python_language_index
    cdef const godot_gdnative_ext_nativescript_1_1_api_struct *ns11api = nativescript_1_1_api

    print("Python language index", i)

    ns11api.godot_nativescript_set_global_type_tag(i, "Object", <const void *>Object)
    ns11api.godot_nativescript_set_global_type_tag(i, "Node", <const void *>Node)
    ns11api.godot_nativescript_set_global_type_tag(i, "CanvasItem", <const void *>CanvasItem)
    ns11api.godot_nativescript_set_global_type_tag(i, "Node2D", <const void *>Node2D)
    ns11api.godot_nativescript_set_global_type_tag(i, "Sprite", <const void *>Sprite)
