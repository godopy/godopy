import math
import numpy as np
from typing import Dict, List

import godot as gd
from godot import classdb, singletons as gds
from godot.types import *

try:
    # Godot must be compile with a custom module and updated api must be dumped
    # and used to build GodoPy for extended tests to work
    GDExtensionTestCase = classdb.GDExtensionTestCase
except AttributeError:
    GDExtensionTestCase = None

if GDExtensionTestCase:
    class ExtendedTestObject(gd.Class, inherits=GDExtensionTestCase):
        def __init__(self) -> None:
            self.arg1 = None
            self.arg2 = None
            self.arg3 = None
            self.arg4 = None
            self.arg5 = None
            self.arg6 = None
            self.arg7 = None
            self.arg8 = None
            self.arg9 = None
            self.argA = None

            self.gdscript = None

        def set_gdscript_instance(self, gdscript: gd.classdb.Node) -> None:
            self.gdscript = gdscript

        def _atomic_args(self, arg1: bool, arg2: int, arg3: int, arg4: float,
                         arg5: float, arg6: str) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            self.arg4 = arg4
            self.arg5 = arg5
            self.arg6 = arg6

        def _math_args_1(self, arg1: Vector2, arg2: Vector2i, arg3: Rect2,
                         arg4: Rect2i, arg5: Vector3, arg6: Vector3i,
                         arg7: Transform2D, arg8: Vector4, arg9: Vector4i) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            self.arg4 = arg4
            self.arg5 = arg5
            self.arg6 = arg6
            self.arg7 = arg7
            self.arg8 = arg8
            self.arg9 = arg9

        def _math_args_2(self, arg1: Plane, arg2: Quaternion, arg3: AABB,
                         arg4: Basis, arg5: Transform3D, arg6: Projection,
                         arg7: Color) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            self.arg4 = arg4
            self.arg5 = arg5
            self.arg6 = arg6
            self.arg7 = arg7

        def _misc_args(self,  arg1: StringName, arg2: NodePath, arg3: RID,
                      arg4: Object, arg5: Callable, arg6: Signal,
                      arg7: Dict, arg8: List) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            self.arg4 = arg4
            self.arg5 = arg5
            self.arg6 = arg6
            self.arg7 = arg7
            self.arg8 = arg8

        def _packed_array_args(self,  arg1: PackedByteArray,
                               arg2: PackedInt32Array, arg3: PackedInt64Array,
                               arg4: PackedFloat32Array, arg5: PackedFloat64Array,
                               arg6: PackedStringArray, arg7: PackedVector2Array,
                               arg8: PackedVector3Array, arg9: PackedColorArray,
                               argA: PackedVector4Array) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            self.arg4 = arg4
            self.arg5 = arg5
            self.arg6 = arg6
            self.arg7 = arg7
            self.arg8 = arg8
            self.arg9 = arg9
            self.argA = argA

        def _get_bool(self) -> bool:
            return True

        def _get_int32(self) -> int:
            return np.int32(42)

        def _get_int64(self) -> int:
            return 420

        def _get_float32(self) -> float:
            return np.float32(math.pi)

        def _get_float64(self) -> float:
            return math.tau

        def _get_string(self) -> str:
            return 'Godot'

        def _get_vector2(self) -> Vector2:
            return Vector2(2.5, 5)

        def _get_vector2i(self) -> Vector2i:
            return Vector2i(5, 10)

        def _get_rect2(self) -> Rect2:
            return Rect2(0, 1, 100, 200)

        def _get_rect2i(self) -> Rect2i:
            return Rect2i(0, 1, 100, 200)

        def _get_vector3(self) -> Vector3:
            return Vector3(2.5, 5, 10)

        def _get_vector3i(self) -> Vector3i:
            return Vector3i(5, 10, 20)

        def _get_transform2d(self) -> Transform2D:
            return Transform2D(Vector2(1, 2), Vector2(3, 4), Vector2(5, 6))

        def _get_vector4(self) -> Vector4:
            return Vector4(2.5, 5, 10, 20)

        def _get_vector4i(self) -> Vector4i:
            return Vector4i(5, 10, 20, 40)

        def _get_plane(self) -> Plane:
            return Plane(5, 10, 20, 40)

        def _get_quaternion(self) -> Quaternion:
            return Quaternion(5, 10, 20, 40)

        def _get_aabb(self) -> AABB:
            return AABB(5, 10, 20, 40, 50, 60)

        def _get_basis(self) -> Basis:
            return Basis(1, 2, 3, 4, 5, 6, 7, 8, 9)

        def _get_transform3d(self) -> Transform3D:
            return Transform3D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

        def _get_projection(self) -> Projection:
            return Projection(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

        def _get_color(self) -> Color:
            return Color(0.2, 0.4, 0.5, 0.75)

        def _get_string_name(self) -> StringName:
            return StringName("GodoPy")

        def _get_node_path(self) -> NodePath:
            return NodePath("test/resource.py")

        def _get_rid(self) -> RID:
            return self.gdscript.call('get_resource').get_rid()

        def _get_object(self) -> Object:
            return self.gdscript

        def _get_callable(self) -> Callable:
            return self.gdscript.call('get_callable')

        def _get_signal(self) -> Signal:
            return self.gdscript.call('get_signal')

        def _get_dictionary(self) -> Dict:
            return {"hello": "World!", 2: (4, 5, [6, 7, 'a'])}

        def _get_array(self) -> List:
            return ["Hello", 1, 3.1, "World", 2.0]

        def _get_packed_byte_array(self) -> PackedByteArray:
            return as_packed_byte_array([2, 3, 4, 5, 1, 7])

        def _get_packed_int32_array(self) -> PackedInt32Array:
            return as_packed_int32_array([2, 3, 4, 5, 1.7, 7])

        def _get_packed_int64_array(self) -> PackedInt64Array:
            return as_packed_int64_array([2, 3, 4.2, 5, 1, 7])

        def _get_packed_float32_array(self) -> PackedFloat32Array:
            return as_packed_float32_array([2, 3, 4, 5, 1.7, 7])

        def _get_packed_float64_array(self) -> PackedFloat64Array:
            return as_packed_float64_array([2, 3, 4.2, 5, 1, 7])

        def _get_packed_string_array(self) -> List[str]:
            return ["Hello", "World", "!"]

        def _get_packed_vector2_array(self) -> PackedVector2Array:
            return as_packed_vector2_array([[2, 3], [4, 5], [1, 7]])

        def _get_packed_vector3_array(self) -> PackedVector3Array:
            return as_packed_vector3_array([[2, 3, 4], [9, 5, 6], [1, 7, 8]])

        def _get_packed_color_array(self) -> PackedColorArray:
            return as_packed_color_array([[.2, .3, .4], [.9, .5, .6, 0.75], [.1, .7, .8, .1]])

        def _get_packed_vector4_array(self) -> PackedVector4Array:
            return as_packed_vector4_array([[2, 3, 4, 8], [5, 1, 7, 9]])


else:
    ExtendedTestObject = None


class Example(gd.Class, inherits=classdb.Control):
    def __init__(self):
        self.entered_tree = False
        self.exited_tree = False

    def _enter_tree(self):
        self.entered_tree = True

    def _exit_tree(self):
        self.exited_tree = True


class TestResource(gd.Class, inherits=classdb.Resource):
    def __init__(self):
        self.arg1 = None
        self.arg2 = None
        self.arg3 = None
        self.arg4 = None
        self.arg5 = None
        self.arg6 = None
        self.arg7 = None
        self.arg8 = None
        self.arg9 = None
        self.argA = None

    @classdb.bind_method
    def atomic_args(self, arg1: bool, arg2: int, arg3: float, arg4: str):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4

    @classdb.bind_method
    def math_args_1(self, arg1: Vector2, arg2: Vector2i, arg3: Rect2,
                    arg4: Rect2i, arg5: Vector3, arg6: Vector3i,
                    arg7: Transform2D, arg8: Vector4, arg9: Vector4i):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.arg5 = arg5
        self.arg6 = arg6
        self.arg7 = arg7
        self.arg8 = arg8
        self.arg9 = arg9

    @classdb.bind_method
    def math_args_2(self, arg1: Plane, arg2: Quaternion, arg3: AABB,
                    arg4: Basis, arg5: Transform3D, arg6: Projection,
                    arg7: Color):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.arg5 = arg5
        self.arg6 = arg6
        self.arg7 = arg7

    @classdb.bind_method
    def misc_args(self,  arg1: StringName, arg2: NodePath, arg3: RID,
                  arg4: Object, arg5: Callable, arg6: Signal,
                  arg7: Dict, arg8: List):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.arg5 = arg5
        self.arg6 = arg6
        self.arg7 = arg7
        self.arg8 = arg8

    @classdb.bind_method
    def packed_array_args(self,  arg1: PackedByteArray,
                          arg2: PackedInt32Array, arg3: PackedInt64Array,
                          arg4: PackedFloat32Array, arg5: PackedFloat64Array,
                          arg6: PackedStringArray, arg7: PackedVector2Array,
                          arg8: PackedVector3Array, arg9: PackedColorArray,
                          argA: PackedVector4Array):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.arg4 = arg4
        self.arg5 = arg5
        self.arg6 = arg6
        self.arg7 = arg7
        self.arg8 = arg8
        self.arg9 = arg9
        self.argA = argA

    @classdb.bind_method
    def bool_ret(self) -> bool:
        return True
    
    @classdb.bind_method
    def int_ret(self) -> int:
        return 42

    @classdb.bind_method
    def float_ret(self) -> float:
        return math.tau

    @classdb.bind_method
    def string_ret(self) -> str:
        return 'GodoPy'

    @classdb.bind_method
    def vector2_ret(self) -> Vector2:
        return Vector2(2.5, 5)

    @classdb.bind_method
    def vector2i_ret(self) -> Vector2i:
        return Vector2i(5, 10)

    @classdb.bind_method
    def rect2_ret(self) -> Rect2:
        return Rect2(0, 1, 100, 200)

    @classdb.bind_method
    def rect2i_ret(self) -> Rect2i:
        return Rect2i(0, 1, 100, 200)

    @classdb.bind_method
    def vector3_ret(self) -> Vector3:
        return Vector3(2.5, 5, 10)

    @classdb.bind_method
    def vector3i_ret(self) -> Vector3i:
        return Vector3i(5, 10, 20)

    @classdb.bind_method
    def transform2d_ret(self) -> Transform2D:
        return Transform2D(Vector2(1, 2), Vector2(3, 4), Vector2(5, 6))

    @classdb.bind_method
    def vector4_ret(self) -> Vector4:
        return Vector4(2.5, 5, 10, 20)

    @classdb.bind_method
    def vector4i_ret(self) -> Vector4i:
        return Vector4i(5, 10, 20, 40)

    @classdb.bind_method
    def plane_ret(self) -> Plane:
        return Plane(5, 10, 20, 40)

    @classdb.bind_method
    def quaternion_ret(self) -> Quaternion:
        return Quaternion(5, 10, 20, 40)

    @classdb.bind_method
    def aabb_ret(self) -> AABB:
        return AABB(5, 10, 20, 40, 50, 60)

    @classdb.bind_method
    def basis_ret(self) -> Basis:
        return Basis(1, 2, 3, 4, 5, 6, 7, 8, 9)

    @classdb.bind_method
    def transform3d_ret(self) -> Transform3D:
        return Transform3D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    @classdb.bind_method
    def projection_ret(self) -> Projection:
        return Projection(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    @classdb.bind_method
    def color_ret(self) -> Color:
        return Color(0.2, 0.4, 0.5, 0.75)

    @classdb.bind_method
    def string_name_ret(self) -> StringName:
        return StringName("GodoPy")

    @classdb.bind_method
    def node_path_ret(self) -> NodePath:
        return NodePath("test/resource.py")

    @classdb.bind_method
    def rid_ret(self) -> RID:
        return self.get_rid()

    @classdb.bind_method
    def object_ret(self) -> Object:
        return self

    @classdb.bind_method
    def callable_ret(self) -> Callable:
        return lambda x: x + 1

    @classdb.bind_method
    def signal_ret(self) -> Signal:
        return None  # TODO

    @classdb.bind_method
    def dictionary_ret(self) -> Dict:
        return {"hello": 1, "world": 2.0, "!": ("!!!", [4, 5])}

    @classdb.bind_method
    def array_ret(self) -> List:
        return [5, "a", 3.0, 40]

    @classdb.bind_method
    def packed_byte_array_ret(self) -> PackedByteArray:
        return as_packed_byte_array([2, 3, 4, 5, 1, 7])

    @classdb.bind_method
    def packed_int32_array_ret(self) -> PackedInt32Array:
        return as_packed_int32_array([2, 3, 4, 5, 1.7, 7])

    @classdb.bind_method
    def packed_int64_array_ret(self) -> PackedInt64Array:
        return as_packed_int64_array([2, 3, 4.2, 5, 1, 7])

    @classdb.bind_method
    def packed_float32_array_ret(self) -> PackedFloat32Array:
        return as_packed_float32_array([2, 3, 4, 5, 1.7, 7])

    @classdb.bind_method
    def packed_float64_array_ret(self) -> PackedFloat64Array:
        return as_packed_float64_array([2, 3, 4.2, 5, 1, 7])

    @classdb.bind_method
    def packed_string_array_ret(self) -> List[str]:
        return ["Hello", "World", "!"]

    @classdb.bind_method
    def packed_vector2_array_ret(self) -> PackedVector2Array:
        return as_packed_vector2_array([[2, 3], [4, 5], [1, 7]])

    @classdb.bind_method
    def packed_vector3_array_ret(self) -> PackedVector3Array:
        return as_packed_vector3_array([[2, 3, 4], [9, 5, 6], [1, 7, 8]])

    @classdb.bind_method
    def packed_color_array_ret(self) -> PackedColorArray:
        return as_packed_color_array([[.2, .3, .4], [.9, .5, .6, 0.75], [.1, .7, .8, .1]])

    @classdb.bind_method
    def packed_vector4_array_ret(self) -> PackedVector4Array:
        return as_packed_vector4_array([[2, 3, 4, 10], [9, 5, 6, 11], [1, 7, 8, 12]])
