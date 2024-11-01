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
            self._ids = []

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

        def _misc_args(self, arg1: StringName, arg2: NodePath, arg3: RID,
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

        def _packed_array_args(self, arg1: PackedByteArray,
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

        def _other_args_1(self, arg1: Variant, arg2: Pointer, arg3: Pointer,
                          arg4: Pointer, arg5: Pointer, arg6: Pointer,
                          arg7: Pointer, arg8: Pointer, arg9: Pointer) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            self.arg4 = arg4
            self.arg5 = arg5
            self.arg6 = arg6
            self.arg7 = arg7
            self.arg8 = arg8
            self.arg9 = arg9

        def _other_args_2(self, arg1: AudioFrame, arg2: CaretInfo, arg3: Glyph,
                          arg4: ObjectID) -> None:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
            # Special case: ObjectID pointers are passed as void*
            self.arg4 = as_object_id(arg4)

        def _other_args_3(self, arg1: PhysicsServer2DExtensionMotionResult,
                          arg2: PhysicsServer2DExtensionRayResult,
                          arg3: PhysicsServer2DExtensionShapeRestInfo,
                          arg4: PhysicsServer2DExtensionShapeResult,
                          arg5: PhysicsServer3DExtensionMotionCollision,
                          arg6: PhysicsServer3DExtensionMotionResult,
                          arg7: PhysicsServer3DExtensionRayResult,
                          arg8: PhysicsServer3DExtensionShapeRestInfo,
                          arg9: PhysicsServer3DExtensionShapeResult,
                          argA: ScriptLanguageExtensionProfilingInfo):
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

        def _get_variant(self) -> Variant:
            obj = object()
            self._ids.append(id(obj))
            return obj  # OBJECT Variant Type representing PythonObject

        def _get_pointer1(self) -> Pointer:
            arr = np.array([1])
            self._arr1 = arr  # ensure array will be alive during tests
            ptr = Pointer.from_int64_array(arr)
            self._ids.append(ptr.pointer_id())
            return ptr

        def _get_pointer2(self) -> Pointer:
            arr = np.array([2])
            self._arr2 = arr  # ensure array will be alive during tests
            ptr = Pointer.from_int64_array(arr)
            self._ids.append(ptr.pointer_id())
            return ptr

        def _get_uint8_pointer1(self) -> Pointer:
            arr = np.arange(1, 9, dtype=np.uint8)
            self._arr3 = arr  # ensure array will be alive during tests
            ptr = Pointer.from_uint8_array(arr)
            self._ids.append(ptr.pointer_id())
            return ptr

        def _get_uint8_pointer2(self) -> Pointer:
            arr = np.arange(9, 17, dtype=np.uint8)
            self._arr4 = arr  # ensure array will be alive during tests
            ptr = Pointer.from_uint8_array(arr)
            self._ids.append(ptr.pointer_id())
            return ptr

        def _get_uint8_pointer3(self) -> Pointer:
            arr = np.arange(17, 25, dtype=np.uint8)
            self._arr5 = arr  # ensure array will be alive during tests
            ptr = Pointer.from_uint8_array(arr)
            self._ids.append(ptr.pointer_id())
            return ptr

        def _get_uint8_pointer4(self) -> Pointer:
            arr = np.arange(25, 33, dtype=np.uint8)
            self._arr6 = arr  # ensure array will be alive during tests
            ptr = Pointer.from_uint8_array(arr)
            self._ids.append(ptr.pointer_id())
            return ptr

        def _get_audio_frame(self) -> AudioFrame:
            self._af = AudioFrame(1, 2)
            return self._af

        def _get_caret_info(self) -> CaretInfo:
            self._ci = CaretInfo((1, 2, 3, 4), (5, 6, 7, 8), 9, 10)
            return self._ci

        def _get_glyph(self) -> Glyph:
            self._g = Glyph(start=1, end=5)
            return self._g

        def _get_object_id(self) -> ObjectID:
            self._oi = ObjectID(123)
            return self._oi

        def _get_ps2d_motion_result(self) -> PhysicsServer2DExtensionMotionResult:
            self._mr2d = PhysicsServer2DExtensionMotionResult(
                (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), 1.0, 2.0, 3.0, 4, 5,
                RID(), 6
            )
            return self._mr2d

        def _get_ps2d_ray_result(self) -> PhysicsServer2DExtensionRayResult:
            self._rr2d = PhysicsServer2DExtensionRayResult((1, 2), (3, 4), RID(), 1, self, 2)
            return self._rr2d

        def _get_ps2d_shape_rest_info(self) -> PhysicsServer2DExtensionShapeRestInfo:
            self._sri2d = PhysicsServer2DExtensionShapeRestInfo((1, 2), (3, 4), RID(), 1, 2, (5, 6))
            return self._sri2d

        def _get_ps2d_shape_result(self) -> PhysicsServer2DExtensionShapeResult:
            self._sr2d = PhysicsServer2DExtensionShapeResult(RID(), 1, self, 2)
            return self._sr2d

        def _get_ps3d_motion_collision(self) -> PhysicsServer3DExtensionMotionCollision:
            self._mc3d = PhysicsServer3DExtensionMotionCollision(
                (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12),
                13., 14, 15, RID(), 16
            )
            return self._mc3d

        def _get_ps3d_motion_result(self) -> PhysicsServer3DExtensionMotionResult:
            self._mr3d = PhysicsServer3DExtensionMotionResult(
                [1., 2., 3.], [4., 5., 6.], 7., 8., 9.,
                [
                    PhysicsServer3DExtensionMotionCollision(
                        (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12),
                        13., 14, 15, RID(), 16
                    )
                ]
            )
            return self._mr3d

        def _get_ps3d_ray_result(self) -> PhysicsServer3DExtensionRayResult:
            self._rr3d = PhysicsServer3DExtensionRayResult((1, 2, 3), (4, 5, 6), RID(), 7, self, 8, 9)
            return self._rr3d

        def _get_ps3d_shape_rest_info(self) -> PhysicsServer3DExtensionShapeRestInfo:
            self._sri3d =  PhysicsServer3DExtensionShapeRestInfo((1, 2, 3), (4, 5, 6), RID(), 7, 8, (9, 10, 11))
            return self._sri3d

        def _get_ps3d_shape_result(self) -> PhysicsServer3DExtensionShapeResult:
            self._sr3d = PhysicsServer3DExtensionShapeResult(RID(), 1, self, 2)
            return self._sr3d

        def _get_script_language_profiling_info(self) -> ScriptLanguageExtensionProfilingInfo:
            self._slpi = ScriptLanguageExtensionProfilingInfo("test", 1, 2, 3)
            return self._slpi

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
