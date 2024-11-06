import math
import numpy as np
from typing import List

import godot as gd
from godot.types import *

from ._base import BaseTestCase
from classes import ExtendedTestObject

__all__ = []

if ExtendedTestObject is not None:
    __all__ += ['TestCaseArgTypesExtended']

    class TestCaseArgTypesExtended(BaseTestCase):
        def _assert_atomic_types(self, t: ExtendedTestObject) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                t.get_bool(),
                t.get_int32(),
                t.get_int64(),
                t.get_float32(),
                t.get_float64(),
                t.get_string()
            ]

            self.assertIsInstance(args[0], bool)
            self.assertEqual(args[0], True)
            self.assertIsInstance(args[1], int)
            self.assertEqual(args[1], 42)
            self.assertIsInstance(args[2], int)
            self.assertEqual(args[2], 420)
            self.assertIsInstance(args[3], float)
            self.assertEqual(args[3], np.float32(math.pi))
            self.assertIsInstance(args[4], float)
            self.assertEqual(args[4], math.tau)
            self.assertIsInstance(args[5], str)
            self.assertEqual(args[5], 'Godot')


        def test_atomic_type_args_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to builtin Engine's types
            t.atomic_args(
                True,
                42,
                420,
                math.pi,
                math.tau,
                "Godot"
            )

            self._assert_atomic_types(t)

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.atomic_args_virtual_call()

            self.assertIsInstance(t.arg1, bool)
            self.assertEqual(t.arg1, True)
            self.assertIsInstance(t.arg2, int)
            self.assertEqual(t.arg2, 42)
            self.assertIsInstance(t.arg3, int)
            self.assertEqual(t.arg3, 420)
            self.assertIsInstance(t.arg4, float)
            self.assertEqual(t.arg4, np.float32(math.pi))
            self.assertIsInstance(t.arg5, float)
            self.assertEqual(t.arg5, math.tau)
            self.assertIsInstance(t.arg6, str)
            self.assertEqual(t.arg6, 'Godot')


        def test_atomic_type_return_from_python_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make virtual calls from the Engine to Python
            # Cover return value conversion from Python to builtin Engine's types
            t.get_bool_virtual_call()
            t.get_int32_virtual_call()
            t.get_int64_virtual_call()
            t.get_float32_virtual_call()
            t.get_float64_virtual_call()
            t.get_string_virtual_call()

            self._assert_atomic_types(t)


        def _assert_math_types_1(self, t: ExtendedTestObject) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                t.get_vector2(),
                t.get_vector2i(),
                t.get_rect2(),
                t.get_rect2i(),
                t.get_vector3(),
                t.get_vector3i(),
                t.get_transform2d(),
                t.get_vector4(),
                t.get_vector4i()
            ]

            self.assertIsInstance(args[0], Vector2)
            self.assertEqual(args[0].dtype, np.dtype('float32'))
            self.assertEqual(args[0].tolist(), [2.5, 5.])

            self.assertIsInstance(args[1], Vector2i)
            self.assertEqual(args[1].dtype, np.dtype('int32'))
            self.assertEqual(args[1].tolist(), [5, 10])

            self.assertIsInstance(args[2], Rect2)
            self.assertEqual(args[2].dtype, np.dtype('float32'))
            self.assertEqual(args[2].tolist(), [0., 1., 100., 200.])

            self.assertIsInstance(args[3], Rect2i)
            self.assertEqual(args[3].dtype, np.dtype('int32'))
            self.assertEqual(args[3].tolist(), [0, 1, 100, 200])

            self.assertIsInstance(args[4], Vector3)
            self.assertEqual(args[4].dtype, np.dtype('float32'))
            self.assertEqual(args[4].tolist(), [2.5, 5., 10.])

            self.assertIsInstance(args[5], Vector3i)
            self.assertEqual(args[5].dtype, np.dtype('int32'))
            self.assertEqual(args[5].tolist(), [5, 10, 20])

            self.assertIsInstance(args[6], Transform2D)
            self.assertEqual(args[6].dtype, np.dtype('float32'))
            self.assertEqual(args[6].tolist(), [[1., 2.], [3., 4.], [5., 6.]])

            self.assertIsInstance(args[7], Vector4)
            self.assertEqual(args[7].dtype, np.dtype('float32'))
            self.assertEqual(args[7].tolist(), [2.5, 5., 10., 20.])

            self.assertIsInstance(args[8], Vector4i)
            self.assertEqual(args[8].dtype, np.dtype('int32'))
            self.assertEqual(args[8].tolist(), [5, 10, 20, 40])


        def _assert_math_types_2(self, t: ExtendedTestObject) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                t.get_plane(),
                t.get_quaternion(),
                t.get_aabb(),
                t.get_basis(),
                t.get_transform3d(),
                t.get_projection(),
                t.get_color(),
            ]

            self.assertIsInstance(args[0], Plane)
            self.assertEqual(args[0].dtype, np.dtype('float32'))
            self.assertEqual(args[0].tolist(), [5, 10, 20, 40])

            self.assertIsInstance(args[1], Quaternion)
            self.assertEqual(args[1].dtype, np.dtype('float32'))
            self.assertEqual(args[1].tolist(), [5, 10, 20, 40])

            self.assertIsInstance(args[2], AABB)
            self.assertEqual(args[2].dtype, np.dtype('float32'))
            self.assertEqual(args[2].tolist(), [[5, 10, 20], [40, 50, 60]])

            self.assertIsInstance(args[3], Basis)
            self.assertEqual(args[3].dtype, np.dtype('float32'))
            self.assertEqual(args[3].tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

            self.assertIsInstance(args[4], Transform3D)
            self.assertEqual(args[4].dtype, np.dtype('float32'))
            self.assertEqual(args[4].tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

            self.assertIsInstance(args[5], Projection)
            self.assertEqual(args[5].dtype, np.dtype('float32'))
            self.assertEqual(args[5].tolist(), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

            self.assertIsInstance(args[6], Color)
            self.assertEqual(args[6].dtype, np.dtype('float32'))
            # Without converting to float32 numbers would differ due to the lost precision
            self.assertEqual(args[6].tolist(), [np.float32(n) for n in (0.2, 0.4, 0.5, 0.75)])

        def test_math_type_args_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to builtin Engine's types
            t.math_args_1([2.5, 5.], [5, 10], [0., 1., 100., 200.], [0, 1, 100, 200],
                          [2.5, 5., 10.], [5, 10, 20], [[1., 2.], [3., 4.], [5., 6.]],
                          [2.5, 5., 10., 20.], [5, 10, 20, 40])

            self._assert_math_types_1(t)

            t.math_args_2(
                Plane(5, 10, 20, 40),
                Quaternion(5, 10, 20, 40),
                AABB([5, 10, 20], [40, 50, 60]),
                Basis([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                Transform3D([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]),
                Projection([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]),
                Color(0.2, 0.4, 0.5, 0.75)
            )

            self._assert_math_types_2(t)

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.math_args_1_virtual_call()

            self.assertIsInstance(t.arg1, Vector2)
            self.assertEqual(t.arg1.dtype, np.dtype('float32'))
            self.assertEqual(t.arg1.tolist(), [2.5, 5.])

            self.assertIsInstance(t.arg2, Vector2i)
            self.assertEqual(t.arg2.dtype, np.dtype('int32'))
            self.assertEqual(t.arg2.tolist(), [5, 10])

            self.assertIsInstance(t.arg3, Rect2)
            self.assertEqual(t.arg3.dtype, np.dtype('float32'))
            self.assertEqual(t.arg3.tolist(), [0., 1., 100., 200.])

            self.assertIsInstance(t.arg4, Rect2i)
            self.assertEqual(t.arg4.dtype, np.dtype('int32'))
            self.assertEqual(t.arg4.tolist(), [0, 1, 100, 200])

            self.assertIsInstance(t.arg5, Vector3)
            self.assertEqual(t.arg5.dtype, np.dtype('float32'))
            self.assertEqual(t.arg5.tolist(), [2.5, 5., 10.])

            self.assertIsInstance(t.arg6, Vector3i)
            self.assertEqual(t.arg6.dtype, np.dtype('int32'))
            self.assertEqual(t.arg6.tolist(), [5, 10, 20])

            self.assertIsInstance(t.arg7, Transform2D)
            self.assertEqual(t.arg7.dtype, np.dtype('float32'))
            self.assertEqual(t.arg7.tolist(), [[1., 2.], [3., 4.], [5., 6.]])

            self.assertIsInstance(t.arg8, Vector4)
            self.assertEqual(t.arg8.dtype, np.dtype('float32'))
            self.assertEqual(t.arg8.tolist(), [2.5, 5., 10., 20.])

            self.assertIsInstance(t.arg9, Vector4i)
            self.assertEqual(t.arg9.dtype, np.dtype('int32'))
            self.assertEqual(t.arg9.tolist(), [5, 10, 20, 40])


            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.math_args_2_virtual_call()

            self.assertIsInstance(t.arg1, Plane)
            self.assertEqual(t.arg1.dtype, np.dtype('float32'))
            self.assertEqual(t.arg1.tolist(), [5, 10, 20, 40])

            self.assertIsInstance(t.arg2, Quaternion)
            self.assertEqual(t.arg2.dtype, np.dtype('float32'))
            self.assertEqual(t.arg2.tolist(), [5, 10, 20, 40])

            self.assertIsInstance(t.arg3, AABB)
            self.assertEqual(t.arg3.dtype, np.dtype('float32'))
            self.assertEqual(t.arg3.tolist(), [[5, 10, 20], [40, 50, 60]])

            self.assertIsInstance(t.arg4, Basis)
            self.assertEqual(t.arg4.dtype, np.dtype('float32'))
            self.assertEqual(t.arg4.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

            self.assertIsInstance(t.arg5, Transform3D)
            self.assertEqual(t.arg5.dtype, np.dtype('float32'))
            self.assertEqual(t.arg5.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

            self.assertIsInstance(t.arg6, Projection)
            self.assertEqual(t.arg6.dtype, np.dtype('float32'))
            self.assertEqual(t.arg6.tolist(), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

            self.assertIsInstance(t.arg7, Color)
            self.assertEqual(t.arg7.dtype, np.dtype('float32'))
            # Without converting to float32 numbers would differ due to the lost precision
            self.assertEqual(t.arg7.tolist(), [np.float32(n) for n in (0.2, 0.4, 0.5, 0.75)])


        def test_math_type_return_from_python_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make virtual calls from the Engine to Python
            # Cover return value conversion from Python to builtin Engine's types
            t.get_vector2_virtual_call()
            t.get_vector2i_virtual_call()
            t.get_rect2_virtual_call()
            t.get_rect2i_virtual_call()
            t.get_vector3_virtual_call()
            t.get_vector3i_virtual_call()
            t.get_transform2d_virtual_call()
            t.get_vector4_virtual_call()
            t.get_vector4i_virtual_call()
            t.get_plane_virtual_call()
            t.get_quaternion_virtual_call()
            t.get_aabb_virtual_call()
            t.get_basis_virtual_call()
            t.get_transform3d_virtual_call()
            t.get_projection_virtual_call()
            t.get_color_virtual_call()

            self._assert_math_types_1(t)
            self._assert_math_types_2(t)


        def _assert_misc_types(self, t: ExtendedTestObject, gdscript: gd.classdb.Node) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                t.get_string_name(),
                t.get_node_path(),
                t.get_rid(),
                t.get_object(),
                t.get_callable(),
                t.get_signal(),
                t.get_dictionary(),
                t.get_array(),
            ]
            self.assertIsInstance(args[0], StringName)
            self.assertEqual(args[0], "GodoPy")

            self.assertIsInstance(args[1], NodePath)
            self.assertEqual(str(args[1]), 'test/resource.py')

            self.assertIsInstance(args[2], RID)
            # XXX: gdscript.call_script_method crashes in gdextension_interface_object_call_script_method
            res = gdscript.call('get_resource')
            self.assertEqual(int(args[2]), int(res.get_rid()))

            self.assertIsInstance(args[3], Object)
            self.assertEqual(args[3].__class__.__name__, "Node")

            self.assertIsInstance(args[4], Callable)
            self.assertIsInstance(args[5], Signal)

            self.assertIsInstance(args[6], dict)
            self.assertEqual(args[6], {"hello": "World!", 2: [4, 5, [6, 7, 'a']]})

            self.assertIsInstance(args[7], list)
            self.assertEqual(args[7], ["Hello", 1, float(3.1), "World", 2.0])


        def test_misc_type_args_ptrcalls(self) -> None:
            t = ExtendedTestObject()
            gdscript = self._main.get_node('TestCasesGDScript')

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to builtin Engine's types
            t.misc_args(
                StringName('GodoPy'),
                NodePath('test/resource.py'),
                gdscript.call_script_method('get_resource').get_rid(),
                gdscript,
                gdscript.call_script_method('get_callable'),
                gdscript.call_script_method('get_signal'),
                {"hello": "World!", 2: [4, 5, [6, 7, 'a']]},
                ["Hello", 1, 3.1, "World", 2.0]
            )

            self._assert_misc_types(t, gdscript)

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.misc_args_virtual_call()

            self.assertIsInstance(t.arg1, StringName)
            self.assertEqual(t.arg1, "GodoPy")

            self.assertIsInstance(t.arg2, NodePath)
            self.assertEqual(str(t.arg2), "test/resource.py")

            self.assertIsInstance(t.arg3, RID)
            self.assertEqual(int(t.arg3), int(gdscript.call_script_method('get_resource').get_rid()))

            self.assertIsInstance(t.arg4, Object)
            self.assertEqual(t.arg4, gdscript)

            self.assertIsInstance(t.arg5, Callable)
            self.assertIsInstance(t.arg6, Signal)

            self.assertIsInstance(t.arg7, dict)
            self.assertEqual(t.arg7, {"hello": "World!", 2: [4, 5, [6, 7, 'a']]})

            self.assertIsInstance(t.arg8, list)
            self.assertEqual(t.arg8, ["Hello", 1, 3.1, "World", 2.0])


        def test_misc_type_return_from_python_ptrcalls(self) -> None:
            t = ExtendedTestObject()
            gdscript = self._main.get_node('TestCasesGDScript')

            t.set_gdscript_instance(gdscript)

            # Make virtual calls from the Engine to Python
            # Cover return value conversion from Python to builtin Engine's types
            t.get_string_name_virtual_call()
            t.get_node_path_virtual_call()
            t.get_rid_virtual_call()
            t.get_object_virtual_call()
            t.get_callable_virtual_call()
            t.get_signal_virtual_call()
            t.get_dictionary_virtual_call()
            t.get_array_virtual_call()

            self._assert_misc_types(t, gdscript)


        def _assert_packed_array_types(self, t: ExtendedTestObject) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                t.get_packed_byte_array(),
                t.get_packed_int32_array(),
                t.get_packed_int64_array(),
                t.get_packed_float32_array(),
                t.get_packed_float64_array(),
                t.get_packed_string_array(),
                t.get_packed_vector2_array(),
                t.get_packed_vector3_array(),
                t.get_packed_color_array(),
                t.get_packed_vector4_array(),
            ]

            self.assertIsInstance(args[0], PackedByteArray)
            self.assertEqual(args[0].dtype, np.dtype('uint8'))
            self.assertEqual(args[0].tolist(), [2, 3, 4, 5, 1, 7])

            self.assertIsInstance(args[1], PackedInt32Array)
            self.assertEqual(args[1].dtype, np.dtype('int32'))
            self.assertEqual(args[1].tolist(), [2, 3, 4, 5, 1, 7])

            self.assertIsInstance(args[2], PackedInt64Array)
            self.assertEqual(args[2].dtype, np.dtype('int64'))
            self.assertEqual(args[2].tolist(), [2, 3, 4, 5, 1, 7])

            self.assertIsInstance(args[3], PackedFloat32Array)
            self.assertEqual(args[3].dtype, np.dtype('float32'))
            self.assertEqual(args[3].tolist(), [2., 3., 4., 5., np.float32(1.7), 7.])

            self.assertIsInstance(args[4], PackedFloat64Array)
            self.assertEqual(args[4].dtype, np.dtype('float64'))
            self.assertEqual(args[4].tolist(), [2., 3., 4.2, 5., 1., 7.])

            self.assertIsInstance(args[5], list)
            self.assertEqual(args[5], ['Hello', 'World', '!'])

            self.assertIsInstance(args[6], PackedVector2Array)
            self.assertEqual(args[6].dtype, np.dtype('float32'))
            self.assertEqual(args[6].shape[1], 2)
            self.assertIsInstance(args[6][0], Vector2)
            self.assertEqual(args[6].tolist(), [[2, 3], [4, 5], [1, 7]])

            self.assertIsInstance(args[7], PackedVector3Array)
            self.assertEqual(args[7].dtype, np.dtype('float32'))
            self.assertEqual(args[7].shape[1], 3)
            self.assertIsInstance(args[7][0], Vector3)
            self.assertEqual(args[7].tolist(), [[2, 3, 4], [9, 5, 6], [1, 7, 8]])

            self.assertIsInstance(args[8], PackedColorArray)
            self.assertEqual(args[8].dtype, np.dtype('float32'))
            self.assertEqual(args[8].shape[1], 4)
            self.assertIsInstance(args[8][0], Color)
            self.assertEqual(
                [[np.float32(y) for y in x] for x in args[8].tolist()],
                [[.2, .3, .4, 1.], [.9, .5, .6, 0.75], [.1, .7, .8, .1]]
            )

            self.assertIsInstance(args[9], PackedVector4Array)
            self.assertEqual(args[9].dtype, np.dtype('float32'))
            self.assertEqual(args[9].shape[1], 4)
            self.assertIsInstance(args[9][0], Vector4)
            self.assertEqual(args[9].tolist(), [[2, 3, 4, 8], [5, 1, 7, 9]])


        def test_packed_array_type_args_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to builtin Engine's types
            t.packed_array_args(
                PackedByteArray([2, 3, 4, 5, 1, 7]),
                PackedInt32Array([2, 3, 4, 5, 1, 7]),
                PackedInt64Array([2, 3, 4, 5, 1, 7]),
                PackedFloat32Array([2, 3, 4, 5, 1.7, 7]),
                PackedFloat64Array([2, 3, 4.2, 5, 1, 7]),
                PackedStringArray(['Hello', 'World', '!']),
                PackedVector2Array([[2, 3], [4, 5], [1, 7]]),
                PackedVector3Array([[2, 3, 4], [9, 5, 6], [1, 7, 8]]),
                PackedColorArray([[.2, .3, .4], [.9, .5, .6, 0.75], [.1, .7, .8, .1]]),
                PackedVector4Array([[2, 3, 4, 8], [5, 1, 7, 9]])
            )

            self._assert_packed_array_types(t)

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.packed_array_args_virtual_call()

            self.assertIsInstance(t.arg1, PackedByteArray)
            self.assertEqual(t.arg1.dtype, np.dtype('uint8'))
            self.assertEqual(t.arg1.tolist(), [2, 3, 4, 5, 1, 7])

            self.assertIsInstance(t.arg2, PackedInt32Array)
            self.assertEqual(t.arg2.dtype, np.dtype('int32'))
            self.assertEqual(t.arg2.tolist(), [2, 3, 4, 5, 1, 7])

            self.assertIsInstance(t.arg3, PackedInt64Array)
            self.assertEqual(t.arg3.dtype, np.dtype('int64'))
            self.assertEqual(t.arg3.tolist(), [2, 3, 4, 5, 1, 7])

            self.assertIsInstance(t.arg4, PackedFloat32Array)
            self.assertEqual(t.arg4.dtype, np.dtype('float32'))
            self.assertEqual([np.float32(x) for x in t.arg4.tolist()], [2., 3., 4., 5., 1.7, 7.])

            self.assertIsInstance(t.arg5, PackedFloat64Array)
            self.assertEqual(t.arg5.dtype, np.dtype('float64'))
            self.assertEqual(t.arg5.tolist(), [2., 3., 4.2, 5., 1., 7.])

            self.assertIsInstance(t.arg6, list)
            self.assertEqual(t.arg6, ['Hello', 'World', '!'])

            self.assertIsInstance(t.arg7, PackedVector2Array)
            self.assertEqual(t.arg7.dtype, np.dtype('float32'))
            self.assertEqual(t.arg7.shape[1], 2)
            self.assertIsInstance(t.arg7[0], Vector2)
            self.assertEqual(t.arg7.tolist(), [[2, 3], [4, 5], [1, 7]])

            self.assertIsInstance(t.arg8, PackedVector3Array)
            self.assertEqual(t.arg8.dtype, np.dtype('float32'))
            self.assertEqual(t.arg8.shape[1], 3)
            self.assertIsInstance(t.arg8[0], Vector3)
            self.assertEqual(t.arg8.tolist(), [[2, 3, 4], [9, 5, 6], [1, 7, 8]])

            self.assertIsInstance(t.arg9, PackedColorArray)
            self.assertEqual(t.arg9.dtype, np.dtype('float32'))
            self.assertEqual(t.arg9.shape[1], 4)
            self.assertIsInstance(t.arg9[0], Color)
            self.assertEqual(
                [[np.float32(y) for y in x] for x in t.arg9.tolist()],
                [[.2, .3, .4, 1.], [.9, .5, .6, 0.75], [.1, .7, .8, .1]]
            )

            self.assertIsInstance(t.argA, PackedVector4Array)
            self.assertEqual(t.argA.dtype, np.dtype('float32'))
            self.assertEqual(t.argA.shape[1], 4)
            self.assertIsInstance(t.argA[0], Vector4)
            self.assertEqual(t.argA.tolist(), [[2, 3, 4, 8], [5, 1, 7, 9]])


        def test_packed_array_type_return_from_python_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make virtual calls from the Engine to Python
            # Cover return value conversion from Python to builtin Engine's types
            t.get_packed_byte_array_virtual_call()
            t.get_packed_int32_array_virtual_call()
            t.get_packed_int64_array_virtual_call()
            t.get_packed_float32_array_virtual_call()
            t.get_packed_float64_array_virtual_call()
            t.get_packed_string_array_virtual_call()
            t.get_packed_vector2_array_virtual_call()
            t.get_packed_vector3_array_virtual_call()
            t.get_packed_color_array_virtual_call()
            t.get_packed_vector4_array_virtual_call()

            self._assert_packed_array_types(t)


        def _assert_other_types_1(self, t: ExtendedTestObject, ids: List[int]) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                t.get_variant(),
                t.get_pointer1(),
                t.get_pointer2(),
                t.get_uint8_pointer1(),
                t.get_uint8_pointer2(),
                t.get_uint8_pointer3(),
                t.get_uint8_pointer4()
            ]

            self.assertIsInstance(args[0], gd.classdb.PythonObject)
            self.assertEqual(args[0].call('get_python_object_id'), ids[0])

            self.assertIsInstance(args[1], Pointer)
            self.assertEqual(args[1].pointer_id(), ids[1])

            self.assertIsInstance(args[2], Pointer)
            self.assertEqual(args[2].pointer_id(), ids[2])

            buffer1 = Buffer(args[3], 8)
            self.assertIsInstance(args[3], Pointer)
            # pointer_ids are different because data was copied to C array
            self.assertNotEqual(args[3].pointer_id(), ids[3])
            self.assertEqual(buffer1.as_array().tolist(), [1, 2, 3, 4, 5, 6, 7, 8])

            buffer2 = Buffer(args[4], 8)
            self.assertIsInstance(args[4], Pointer)
            self.assertNotEqual(args[4].pointer_id(), ids[4])
            self.assertEqual(buffer2.as_array().tolist(), [9, 10, 11, 12, 13, 14, 15, 16])

            buffer3 = Buffer(args[5], 8)
            self.assertIsInstance(args[5], Pointer)
            self.assertNotEqual(args[5].pointer_id(), ids[5])
            self.assertEqual(buffer3.as_array().tolist(), [17, 18, 19, 20, 21, 22, 23, 24])

            buffer4 = Buffer(args[6], 8)
            self.assertIsInstance(args[6], Pointer)
            self.assertNotEqual(args[6].pointer_id(), ids[6])
            self.assertEqual(buffer4.as_array().tolist(), [25, 26, 27, 28, 29, 30, 31, 32])


        def _assert_other_types_2(self, t: ExtendedTestObject) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                as_audio_frame(t.get_audio_frame()),
                as_caret_info(t.get_caret_info()),
                as_glyph(t.get_glyph()),
                as_object_id(t.get_object_id()),
            ]

            self.assertIsInstance(args[0], AudioFrame)
            self.assertEqual(args[0].left, 1)

            self.assertIsInstance(args[1], CaretInfo)
            self.assertEqual(args[1].leading_caret.tolist(), [1, 2, 3, 4])

            self.assertIsInstance(args[2], Glyph)
            self.assertEqual(args[2].start, 1)

            self.assertIsInstance(args[3], ObjectID)
            self.assertEqual(args[3].id, 123)


        def _assert_other_types_3(self, t: ExtendedTestObject) -> None:
            # Cover return value conversion from builtin Engine's types to Python
            args = [
                as_physics_server2d_extension_motion_result(t.get_ps2d_motion_result()),
                as_physics_server2d_extension_ray_result(t.get_ps2d_ray_result()),
                as_physics_server2d_extension_shape_rest_info(t.get_ps2d_shape_rest_info()),
                as_physics_server2d_extension_shape_result(t.get_ps2d_shape_result()),
                as_physics_server3d_extension_motion_collision(t.get_ps3d_motion_collision()),
                as_physics_server3d_extension_motion_result(t.get_ps3d_motion_result()),
                as_physics_server3d_extension_ray_result(t.get_ps3d_ray_result()),
                as_physics_server3d_extension_shape_rest_info(t.get_ps3d_shape_rest_info()),
                as_physics_server3d_extension_shape_result(t.get_ps3d_shape_result()),
                as_script_language_extension_profiling_info(t.get_script_language_profiling_info())
            ]


            self.assertIsInstance(args[0], PhysicsServer2DExtensionMotionResult)
            self.assertEqual(args[0].travel.tolist(), [1, 2])

            self.assertIsInstance(args[1], PhysicsServer2DExtensionRayResult)
            self.assertEqual(args[1].position.tolist(), [1, 2])

            self.assertIsInstance(args[2], PhysicsServer2DExtensionShapeRestInfo)
            self.assertEqual(args[2].point.tolist(), [1, 2])

            self.assertIsInstance(args[3], PhysicsServer2DExtensionShapeResult)
            self.assertEqual(args[3].shape, 2)

            self.assertIsInstance(args[4], PhysicsServer3DExtensionMotionCollision)
            self.assertEqual(args[4].position.tolist(), [1., 2., 3.])

            self.assertIsInstance(args[5], PhysicsServer3DExtensionMotionResult)
            self.assertEqual(args[5].travel.tolist(), [1., 2., 3.])

            self.assertIsInstance(args[6], PhysicsServer3DExtensionRayResult)
            self.assertEqual(args[6].position.tolist(), [1., 2., 3.])

            self.assertIsInstance(args[7], PhysicsServer3DExtensionShapeRestInfo)
            self.assertEqual(args[7].point.tolist(), [1., 2., 3.])

            self.assertIsInstance(args[8], PhysicsServer3DExtensionShapeResult)
            self.assertEqual(args[8].shape, 2)

            self.assertIsInstance(args[9], ScriptLanguageExtensionProfilingInfo)
            self.assertEqual(args[9].signature, 'test')


        def test_other_type_args_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            arr_int1 = np.array([1])
            arr_int2 = np.array([2])
            arr_byte1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
            arr_byte2 = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.uint8)
            arr_byte3 = np.array([17, 18, 19, 20, 21, 22, 23, 24], dtype=np.uint8)
            arr_byte4 = np.array([25, 26, 27, 28, 29, 30, 31, 32], dtype=np.uint8)

            obj = object()
            ptr1 = Pointer.from_int64_array(arr_int1)
            ptr2 = Pointer.from_int64_array(arr_int2)
            byte_ptr1 = Pointer.from_uint8_array(arr_byte1)
            byte_ptr2 = Pointer.from_uint8_array(arr_byte2)
            byte_ptr3 = Pointer.from_uint8_array(arr_byte3)
            byte_ptr4 = Pointer.from_uint8_array(arr_byte4)

            args = [
                obj,
                ptr1,
                ptr2,
                byte_ptr1,
                byte_ptr2,
                byte_ptr3,
                byte_ptr4
            ]

            ids = [id(obj)] + [ptr.pointer_id() for ptr in args[1:]]

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to Engine's types
            t.other_args_1(*args)
            self._assert_other_types_1(t, ids)

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to Engine's types
            t.other_args_2(
                AudioFrame(1, 2),
                CaretInfo((1, 2, 3, 4), (5, 6, 7, 8), 9, 10),
                Glyph(start=1, end=5),
                ObjectID(123)
            )
            self._assert_other_types_2(t)

            collision3d = PhysicsServer3DExtensionMotionCollision(
                (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12),
                13., 14, 15, RID(), 16
            )

            # Make a call from Python to the Engine
            # Cover argument conversion from Python to Engine's types
            t.other_args_3(
                PhysicsServer2DExtensionMotionResult((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), 1.0, 2.0, 3.0, 4, 5,
                                                     RID(), 6),
                PhysicsServer2DExtensionRayResult((1, 2), (3, 4), RID(), 1, t, 2),
                PhysicsServer2DExtensionShapeRestInfo((1, 2), (3, 4), RID(), 1, 2, (5, 6)),
                PhysicsServer2DExtensionShapeResult(RID(), 1, t, 2),
                collision3d,
                PhysicsServer3DExtensionMotionResult((1, 2, 3), (4, 5, 6), 7., 8., 9., [collision3d]),
                PhysicsServer3DExtensionRayResult((1, 2, 3), (4, 5, 6), RID(), 7, t, 8, 9),
                PhysicsServer3DExtensionShapeRestInfo((1, 2, 3), (4, 5, 6), RID(), 7, 8, (9, 10, 11)),
                PhysicsServer3DExtensionShapeResult(RID(), 1, t, 2),
                ScriptLanguageExtensionProfilingInfo("test", 1, 2, 3)
            )
            self._assert_other_types_3(t)

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.other_args_1_virtual_call()

            self.assertIsInstance(t.arg1, gd.classdb.PythonObject)
            self.assertEqual(t.arg1.call('get_python_object_id'), ids[0])

            self.assertIsInstance(t.arg2, Pointer)
            self.assertEqual(t.arg2.pointer_id(), ids[1])

            self.assertIsInstance(t.arg3, Pointer)
            self.assertEqual(t.arg3.pointer_id(), ids[2])

            buffer1 = Buffer(t.arg4, 8)
            self.assertIsInstance(t.arg4, Pointer)
            # pointer_ids are different because data was copied to C array
            self.assertNotEqual(t.arg4.pointer_id(), ids[3])
            self.assertEqual(buffer1.as_array().tolist(), [1, 2, 3, 4, 5, 6, 7, 8])

            buffer2 = Buffer(t.arg5, 8)
            self.assertIsInstance(t.arg5, Pointer)
            self.assertNotEqual(t.arg5.pointer_id(), ids[4])
            self.assertEqual(buffer2.as_array().tolist(), [9, 10, 11, 12, 13, 14, 15, 16])

            buffer3 = Buffer(t.arg6, 8)
            self.assertIsInstance(t.arg6, Pointer)
            self.assertNotEqual(t.arg6.pointer_id(), ids[5])
            self.assertEqual(buffer3.as_array().tolist(), [17, 18, 19, 20, 21, 22, 23, 24])

            buffer4 = Buffer(t.arg7, 8)
            self.assertIsInstance(t.arg7, Pointer)
            self.assertNotEqual(t.arg7.pointer_id(), ids[6])
            self.assertEqual(buffer4.as_array().tolist(), [25, 26, 27, 28, 29, 30, 31, 32])

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.other_args_2_virtual_call()

            self.assertIsInstance(t.arg1, AudioFrame)
            self.assertEqual(t.arg1.left, 1)

            self.assertIsInstance(t.arg2, CaretInfo)
            self.assertEqual(t.arg2.leading_caret.tolist(), [1, 2, 3, 4])

            self.assertIsInstance(t.arg3, Glyph)
            self.assertEqual(t.arg3.start, 1)

            self.assertIsInstance(t.arg4, ObjectID)
            self.assertEqual(t.arg4.id, 123)

            # Make a virtual call from the Engine to Python
            # Cover argument conversion from builtin Engine's types to Python
            t.other_args_3_virtual_call()

            self.assertIsInstance(t.arg1, PhysicsServer2DExtensionMotionResult)
            self.assertEqual(t.arg1.travel.tolist(), [1, 2])

            self.assertIsInstance(t.arg2, PhysicsServer2DExtensionRayResult)
            self.assertEqual(t.arg2.position.tolist(), [1, 2])

            self.assertIsInstance(t.arg3, PhysicsServer2DExtensionShapeRestInfo)
            self.assertEqual(t.arg3.point.tolist(), [1, 2])

            self.assertIsInstance(t.arg4, PhysicsServer2DExtensionShapeResult)
            self.assertEqual(t.arg4.shape, 2)

            self.assertIsInstance(t.arg5, PhysicsServer3DExtensionMotionCollision)
            self.assertEqual(t.arg5.position.tolist(), [1, 2, 3])

            self.assertIsInstance(t.arg6, PhysicsServer3DExtensionMotionResult)
            self.assertEqual(t.arg6.travel.tolist(), [1, 2, 3])

            self.assertIsInstance(t.arg7, PhysicsServer3DExtensionRayResult)
            self.assertEqual(t.arg7.position.tolist(), [1, 2, 3])

            self.assertIsInstance(t.arg8, PhysicsServer3DExtensionShapeRestInfo)
            self.assertEqual(t.arg8.point.tolist(), [1, 2, 3])

            self.assertIsInstance(t.arg9, PhysicsServer3DExtensionShapeResult)
            self.assertEqual(t.arg9.shape, 2)

            self.assertIsInstance(t.argA, ScriptLanguageExtensionProfilingInfo)
            self.assertEqual(t.argA.signature, 'test')


        def test_other_type_return_from_python_ptrcalls(self) -> None:
            t = ExtendedTestObject()

            # Make virtual calls from the Engine to Python
            # Cover return value conversion from Python to builtin Engine's types
            t.get_variant_virtual_call()
            t.get_pointer1_virtual_call()
            t.get_pointer2_virtual_call()
            t.get_uint8_pointer1_virtual_call()
            t.get_uint8_pointer2_virtual_call()
            t.get_uint8_pointer3_virtual_call()
            t.get_uint8_pointer4_virtual_call()

            t.get_audio_frame_virtual_call()
            t.get_caret_info_virtual_call()
            t.get_glyph_virtual_call()
            t.get_object_id_virtual_call()

            t.get_ps2d_motion_result_virtual_call()
            t.get_ps2d_ray_result_virtual_call()
            t.get_ps2d_shape_rest_info_virtual_call()
            t.get_ps2d_shape_result_virtual_call()
            t.get_ps3d_motion_collision_virtual_call()
            t.get_ps3d_motion_result_virtual_call()
            t.get_ps3d_ray_result_virtual_call()
            t.get_ps3d_shape_rest_info_virtual_call()
            t.get_ps3d_shape_result_virtual_call()
            t.get_script_language_profiling_info_virtual_call()

            self._assert_other_types_1(t, t._ids)
            self._assert_other_types_2(t)
            self._assert_other_types_3(t)
