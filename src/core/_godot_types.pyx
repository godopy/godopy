"""
Python representations of Godot Variant types

They cannot do much yet.
Used to hold converted Godot data and declare Godot types.
"""

from _godot_type_tuples import *


__all__ = [
    'Nil',

    'bool',
    'int',
    'float',
    'String',

    'Vector2',
    'Vector2i',
    'Rect2',
    'Rect2i',
    'Vector3',
    'Vector3i',
    'Transform2D',
    'Vector4',
    'Vector4i',
    'Plane',
    'Quaternion',
    'AABB',
    'Basis',
    'Transform3D',
    'Projection',

    'Color',
    'StringName',
    'NodePath',
    'RID',
    'Dictionary',
    'Array',

    'PackedByteArray',
    'PackedInt32Array',
    'PackedInt64Array',
    'PackedFloat32Array',
    'PackedFloat64Array',
    'PackedStringArray',
    'PackedVector2Array',
    'PackedVector3Array',
    'PackedColorArray',
    'PackedVector4Array'
]


Nil = None
bool = bool
int = int
float = float
String = str

class Transform2D(tuple):
    pass

class Basis(tuple):
    pass

class Projection(tuple):
    pass

class StringName(str):
    pass

class NodePath(str):
    pass

class RID(int):
    pass

# Object, Callable, Signal are in gdextension module

Dictionary = dict
Array = list

PackedByteArray = bytearray

class PackedInt32Array(tuple):
    pass

class PackedInt64Array(tuple):
    pass

class PackedFloat32Array(tuple):
    pass

class PackedFloat64Array(tuple):
    pass

class PackedStringArray(tuple):
    pass

class PackedVector2Array(tuple):
    pass

class PackedVector3Array(tuple):
    pass

class PackedColorArray(tuple):
    pass

class PackedVector4Array(tuple):
    pass
