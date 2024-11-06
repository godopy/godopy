import gc
import math
import numpy as np
from typing import List

import godot as gd
from godot.types import *

from ._base import BaseTestCase

__all__ = [
    'TestCaseArgTypes'
]

class TestCaseArgTypes(BaseTestCase):
    def test_atomic_type_args_to_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from the Engine to Python
        # Covers argument conversion from Variant to Python and return value convesion from Python to Variant
        gdscript.call_script_method('test_atomic_types_out')

        r = gdscript.call_script_method('get_resource')

        self.assertIsInstance(r.arg1, bool)
        self.assertEqual(r.arg1, True)
        self.assertIsInstance(r.arg2, int)
        self.assertEqual(r.arg2, 42)
        self.assertIsInstance(r.arg3, float)
        self.assertEqual(r.arg3, math.tau)
        self.assertIsInstance(r.arg4, str)
        self.assertEqual(r.arg4, 'GodoPy')
        self.assertIsInstance(r.arg5, str)
        self.assertEqual(r.arg5, '⚠️')
        self.assertIsInstance(r.arg6, str)
        self.assertEqual(r.arg6, 'GodoPy')
        self.assertIsInstance(r.arg7, str)
        self.assertEqual(r.arg7, '⚠️')


    def test_atomic_type_args_from_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from Python to the Engine
        # Covers argument conversion from Python to Variant
        gdscript.call_script_method('test_atomic_types_in', True, 42, math.tau, "GodoPy", "⚠️", b"GodoPy", "⚠️".encode('utf-8'))

        # Covers return value conversion from Variant to Python
        args = [
            gdscript.get('m_bool'),
            gdscript.get('m_int'),
            gdscript.get('m_float'),
            gdscript.get('m_string1'),
            gdscript.get('m_string2'),
            gdscript.get('m_string3'),
            gdscript.get('m_string4'),
        ]
        self.assertIsInstance(args[0], bool)
        self.assertEqual(args[0], True)
        self.assertIsInstance(args[1], int)
        self.assertEqual(args[1], 42)
        self.assertIsInstance(args[2], float)
        self.assertEqual(args[2], math.tau)
        self.assertIsInstance(args[3], str)
        self.assertEqual(args[3], 'GodoPy')
        self.assertIsInstance(args[4], str)
        self.assertEqual(args[4], '⚠️')
        self.assertIsInstance(args[5], str)
        self.assertEqual(args[5], 'GodoPy')
        self.assertIsInstance(args[6], str)
        self.assertEqual(args[6], '⚠️')


    def test_math_type_args_to_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from the Engine to Python
        # Covers argument conversion from Variant to Python and return value convesion from Python to Variant
        gdscript.call_script_method('test_math_types_1_out')

        r = gdscript.call_script_method('get_resource')

        self.assertIsInstance(r.arg1, Vector2)
        self.assertEqual(r.arg1.dtype, np.dtype('float32'))
        self.assertEqual(r.arg1.tolist(), [2.5, 5.])

        self.assertIsInstance(r.arg2, Vector2i)
        self.assertEqual(r.arg2.dtype, np.dtype('int32'))
        self.assertEqual(r.arg2.tolist(), [5, 10])

        self.assertIsInstance(r.arg3, Rect2)
        self.assertEqual(r.arg3.dtype, np.dtype('float32'))
        self.assertEqual(r.arg3.tolist(), [0., 1., 100., 200.])

        self.assertIsInstance(r.arg4, Rect2i)
        self.assertEqual(r.arg4.dtype, np.dtype('int32'))
        self.assertEqual(r.arg4.tolist(), [0, 1, 100, 200])

        self.assertIsInstance(r.arg5, Vector3)
        self.assertEqual(r.arg5.dtype, np.dtype('float32'))
        self.assertEqual(r.arg5.tolist(), [2.5, 5., 10.])

        self.assertIsInstance(r.arg6, Vector3i)
        self.assertEqual(r.arg6.dtype, np.dtype('int32'))
        self.assertEqual(r.arg6.tolist(), [5, 10, 20])

        self.assertIsInstance(r.arg7, Transform2D)
        self.assertEqual(r.arg7.dtype, np.dtype('float32'))
        self.assertEqual(r.arg7.tolist(), [[1., 2.], [3., 4.], [5., 6.]])

        self.assertIsInstance(r.arg8, Vector4)
        self.assertEqual(r.arg8.dtype, np.dtype('float32'))
        self.assertEqual(r.arg8.tolist(), [2.5, 5., 10., 20.])

        self.assertIsInstance(r.arg9, Vector4i)
        self.assertEqual(r.arg9.dtype, np.dtype('int32'))
        self.assertEqual(r.arg9.tolist(), [5, 10, 20, 40])

        # Makes a call from the Engine to Python
        # Covers argument conversion from Variant to Python and return value convesion from Python to Variant
        gdscript.call_script_method('test_math_types_2_out')
        r = gdscript.call_script_method('get_resource')

        self.assertIsInstance(r.arg1, Plane)
        self.assertEqual(r.arg1.dtype, np.dtype('float32'))
        self.assertEqual(r.arg1.tolist(), [5, 10, 20, 40])

        self.assertIsInstance(r.arg2, Quaternion)
        self.assertEqual(r.arg2.dtype, np.dtype('float32'))
        self.assertEqual(r.arg2.tolist(), [5, 10, 20, 40])

        self.assertIsInstance(r.arg3, AABB)
        self.assertEqual(r.arg3.dtype, np.dtype('float32'))
        self.assertEqual(r.arg3.tolist(), [[5, 10, 20], [40, 50, 60]])

        self.assertIsInstance(r.arg4, Basis)
        self.assertEqual(r.arg4.dtype, np.dtype('float32'))
        self.assertEqual(r.arg4.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        self.assertIsInstance(r.arg5, Transform3D)
        self.assertEqual(r.arg5.dtype, np.dtype('float32'))
        self.assertEqual(r.arg5.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        self.assertIsInstance(r.arg6, Projection)
        self.assertEqual(r.arg6.dtype, np.dtype('float32'))
        self.assertEqual(r.arg6.tolist(), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

        self.assertIsInstance(r.arg7, Color)
        self.assertEqual(r.arg7.dtype, np.dtype('float32'))
        # Without converting to float32 numbers would differ due to the lost precision
        self.assertEqual(r.arg7.tolist(), [np.float32(n) for n in (0.2, 0.4, 0.5, 0.75)])


    def test_math_type_args_from_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from Python to the Engine
        # Covers argument conversion from Python to Variant
        gdscript.call_script_method(
            'test_math_types_1_in',
            Vector2(2.5, 5.), Vector2i(5, 10),
            Rect2(0., 1., 100., 200.), Rect2i(0, 1, 100, 200),
            Vector3(2.5, 5., 10.), Vector3i(5, 10, 20),
            Transform2D([1., 2.], [3., 4.], [5., 6.]),
            Vector4(2.5, 5., 10., 20.), Vector4i(5, 10, 20, 40)
        )

        # Covers return value conversion from Variant to Python
        args = [
            gdscript.get('m_vector2'),
            gdscript.get('m_vector2i'),
            gdscript.get('m_rect2'),
            gdscript.get('m_rect2i'),
            gdscript.get('m_vector3'),
            gdscript.get('m_vector3i'),
            gdscript.get('m_transform2d'),
            gdscript.get('m_vector4'),
            gdscript.get('m_vector4i'),
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

        # Makes a call from Python to the Engine
        # Covers argument conversion from Python to Variant
        gdscript.call_script_method(
            'test_math_types_2_in',
            Plane(5, 10, 20, 40),
            Quaternion(5, 10, 20, 40),
            AABB([5, 10, 20], [40, 50, 60]),
            Basis([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            Transform3D([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]),
            Projection([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]),
            Color(0.2, 0.4, 0.5, 0.75)
        )

        # Covers return value conversion from Variant to Python
        args = [
            gdscript.get('m_plane'),
            gdscript.get('m_quaternion'),
            gdscript.get('m_aabb'),
            gdscript.get('m_basis'),
            gdscript.get('m_transform3d'),
            gdscript.get('m_projection'),
            gdscript.get('m_color')
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


    def test_misc_type_args_to_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from the Engine to Python
        # Covers argument conversion from Variant to Python and return value convesion from Python to Variant
        gdscript.call_script_method('test_misc_types_out')

        r = gdscript.call_script_method('get_resource')

        self.assertIsInstance(r.arg1, StringName)
        self.assertEqual(r.arg1, "GodoPy")

        self.assertIsInstance(r.arg2, NodePath)
        self.assertEqual(str(r.arg2), "test/resource.py")

        self.assertIsInstance(r.arg3, RID)
        self.assertEqual(int(r.arg3), int(r.get_rid()))

        self.assertIsInstance(r.arg4, Object)
        self.assertEqual(r.arg4, r)

        # TODO: Callables and Signals

        self.assertIsInstance(r.arg7, dict)
        self.assertEqual(r.arg7, {"hello": 1, "world": 2.0, "!": ["!!!", [4, 5]]})

        self.assertIsInstance(r.arg8, list)
        self.assertEqual(r.arg8, [5, "a", 3.0, 40])

        # TODO: Typed arrays

    def test_misc_args_from_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from Python to the Engine
        # Covers argument conversion from Python to Variant
        gdscript.call_script_method(
            'test_misc_types_in',
            StringName('GodoPy'),
            NodePath('test/resource.py'),
            gdscript.call_script_method('get_resource').get_rid(),
            gdscript,
            gdscript.call_script_method('get_callable'),
            gdscript.call_script_method('get_signal'),
            {"hello": "World!"},
            ["Hello", 1, "World", 2.0]
        )

        # Covers return value conversion from Variant to Python
        args = [
            gdscript.get('m_string_name'),
            gdscript.get('m_node_path'),
            gdscript.get('m_rid'),
            gdscript.get('m_object'),
            gdscript.get('m_callable'),
            gdscript.get('m_signal'),
            gdscript.get('m_dictionary'),
            gdscript.get('m_array')
        ]

        self.assertIsInstance(args[0], StringName)
        self.assertEqual(args[0], "GodoPy")

        self.assertIsInstance(args[1], NodePath)
        self.assertEqual(str(args[1]), 'test/resource.py')

        self.assertIsInstance(args[2], RID)
        self.assertEqual(int(args[2]), int(gdscript.call_script_method('get_resource').get_rid()))

        self.assertIsInstance(args[3], Object)
        self.assertEqual(args[3].__class__.__name__, "Node")

        self.assertIsInstance(args[4], Callable)
        self.assertIsInstance(args[5], Signal)

        self.assertIsInstance(args[6], dict)
        self.assertEqual(args[6], {"hello": "World!"})

        self.assertIsInstance(args[7], list)
        self.assertEqual(args[7], ["Hello", 1, "World", 2.0])


    def test_packed_array_type_args_to_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from the Engine to Python
        # Covers argument conversion from Variant to Python and return value convesion from Python to Variant
        gdscript.call_script_method('test_packed_array_types_out')

        r = gdscript.call_script_method('get_resource')

        self.assertIsInstance(r.arg1, PackedByteArray)
        self.assertEqual(r.arg1.dtype, np.dtype('uint8'))
        self.assertEqual(r.arg1.tolist(), [2, 3, 4, 5, 1, 7])

        self.assertIsInstance(r.arg2, PackedInt32Array)
        self.assertEqual(r.arg2.dtype, np.dtype('int32'))
        self.assertEqual(r.arg2.tolist(), [2, 3, 4, 5, 1, 7])

        self.assertIsInstance(r.arg3, PackedInt64Array)
        self.assertEqual(r.arg3.dtype, np.dtype('int64'))
        self.assertEqual(r.arg3.tolist(), [2, 3, 4, 5, 1, 7])

        self.assertIsInstance(r.arg4, PackedFloat32Array)
        self.assertEqual(r.arg4.dtype, np.dtype('float32'))
        self.assertEqual([np.float32(x) for x in r.arg4.tolist()], [2., 3., 4., 5., 1.7, 7.])

        self.assertIsInstance(r.arg5, PackedFloat64Array)
        self.assertEqual(r.arg5.dtype, np.dtype('float64'))
        self.assertEqual(r.arg5.tolist(), [2., 3., 4.2, 5., 1., 7.])

        self.assertIsInstance(r.arg6, list)
        self.assertEqual(r.arg6, ['Hello', 'World', '!'])

        self.assertIsInstance(r.arg7, PackedVector2Array)
        self.assertEqual(r.arg7.dtype, np.dtype('float32'))
        self.assertEqual(r.arg7.shape[1], 2)
        self.assertIsInstance(r.arg7[0], Vector2)
        self.assertEqual(r.arg7.tolist(), [[2, 3], [4, 5], [1, 7]])

        self.assertIsInstance(r.arg8, PackedVector3Array)
        self.assertEqual(r.arg8.dtype, np.dtype('float32'))
        self.assertEqual(r.arg8.shape[1], 3)
        self.assertIsInstance(r.arg8[0], Vector3)
        self.assertEqual(r.arg8.tolist(), [[2, 3, 4], [9, 5, 6], [1, 7, 8]])

        self.assertIsInstance(r.arg9, PackedColorArray)
        self.assertEqual(r.arg9.dtype, np.dtype('float32'))
        self.assertEqual(r.arg9.shape[1], 4)
        self.assertIsInstance(r.arg9[0], Color)
        self.assertEqual(
            [[np.float32(y) for y in x] for x in r.arg9.tolist()],
            [[.2, .3, .4, 1.], [.9, .5, .6, 0.75], [.1, .7, .8, .1]]
        )

        self.assertIsInstance(r.argA, PackedVector4Array)
        self.assertEqual(r.argA.dtype, np.dtype('float32'))
        self.assertEqual(r.argA.shape[1], 4)
        self.assertIsInstance(r.argA[0], Vector4)
        self.assertEqual(r.argA.tolist(), [[2, 3, 4, 10], [9, 5, 6, 11], [1, 7, 8, 12]])


    def test_packed_array_args_from_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from Python to the Engine
        # Covers argument conversion from Python to Variant
        gdscript.call_script_method(
            'test_packed_array_types_in',
            PackedByteArray([2, 3, 4, 5, 1, 7]),
            PackedInt32Array([2, 3, 4, 5, 1, 7]),
            PackedInt64Array([2, 3, 4, 5, 1, 7]),
            PackedFloat32Array([2, 3, 4, 5, 1, 7]),
            PackedFloat64Array([2, 3, 4, 5, 1, 7]),
            PackedStringArray(['Hello', 'World', '!']),
            PackedVector2Array([[2, 3], [4, 5], [1, 7]]),
            PackedVector3Array([[2, 3, 4], [5, 1, 7]]),
            PackedColorArray([[.2, .3, .4], [.9, .5, .6, 0.75], [.1, .7, .8]]),
            PackedVector4Array([[2, 3, 4, 8], [5, 1, 7, 9]])
        )

        # Covers return value conversion from Variant to Python
        args = [
            gdscript.get('m_packed_byte_array'),
            gdscript.get('m_packed_int32_array'),
            gdscript.get('m_packed_int64_array'),
            gdscript.get('m_packed_float32_array'),
            gdscript.get('m_packed_float64_array'),
            gdscript.get('m_packed_string_array'),
            gdscript.get('m_packed_vector2_array'),
            gdscript.get('m_packed_vector3_array'),
            gdscript.get('m_packed_color_array'),
            gdscript.get('m_packed_vector4_array')
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
        self.assertEqual(args[3].tolist(), [2., 3., 4., 5., 1., 7.])

        self.assertIsInstance(args[4], PackedFloat64Array)
        self.assertEqual(args[4].dtype, np.dtype('float64'))
        self.assertEqual(args[4].tolist(), [2., 3., 4., 5., 1., 7.])

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
        self.assertEqual(args[7].tolist(), [[2, 3, 4], [5, 1, 7]])

        self.assertIsInstance(args[8], PackedColorArray)
        self.assertEqual(args[8].dtype, np.dtype('float32'))
        self.assertEqual(args[8].shape[1], 4)
        self.assertIsInstance(args[8][0], Color)
        self.assertEqual(
            [[np.float32(y) for y in x] for x in args[8].tolist()],
            [[.2, .3, .4, 1.], [.9, .5, .6, 0.75], [.1, .7, .8, 1.]]
        )

        self.assertIsInstance(args[9], PackedVector4Array)
        self.assertEqual(args[9].dtype, np.dtype('float32'))
        self.assertEqual(args[9].shape[1], 4)
        self.assertIsInstance(args[9][0], Vector4)
        self.assertEqual(args[9].tolist(), [[2, 3, 4, 8], [5, 1, 7, 9]])


    def test_other_type_args_to_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from the Engine to Python
        # Covers argument conversion from Variant to Python and return value convesion from Python to Variant
        gdscript.call_script_method('test_other_types_1_out')

        r = gdscript.call_script_method('get_resource')

        self.assertIsInstance(r.arg1, float)
        self.assertEqual(r.arg1, 5.0)

        self.assertIsNone(r.arg2)


    def test_other_type_args_from_python_varcalls(self) -> None:
        gdscript = self._main.get_node('TestCasesGDScript')

        # Makes a call from Python to the Engine
        # Covers argument conversion from Python to Variant
        gdscript.call_script_method('test_other_types_1_in', 5.0, None)

        # Covers return value conversion from Variant to Python
        args = [
            gdscript.get('m_variant'),
            gdscript.get('m_pointer'),
        ]

        self.assertIsInstance(args[0], float)
        self.assertEqual(args[0], 5.0)

        self.assertIsNone(args[1])
