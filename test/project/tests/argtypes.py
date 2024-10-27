import math
import numpy as np

from godot import types, classdb

from ._base import BaseTestCase

__all__ = [
    'TestCaseArgTypes'
]

try:
    GDExtensionTestCase = classdb.GDExtensionTestCase
except AttributeError:
    GDExtensionTestCase = None


if GDExtensionTestCase is not None:
    __all__ += ['TestCaseArgTypesExtended']


    class TestCaseArgTypesExtended(BaseTestCase):
        def test_atomic_types(self):
            t = GDExtensionTestCase()
            t.atomic_args(True, 42, 420, math.pi, math.tau, "Godot")

            self.assertIsInstance(t.get_bool(), bool)
            self.assertEqual(t.get_bool(), True)
            self.assertIsInstance(t.get_int32(), int)
            self.assertEqual(t.get_int32(), 42)
            self.assertIsInstance(t.get_int64(), int)
            self.assertEqual(t.get_int64(), 420)
            self.assertIsInstance(t.get_float32(), float)
            self.assertEqual(t.get_float32(), np.float32(math.pi))
            self.assertIsInstance(t.get_float64(), float)
            self.assertEqual(t.get_float64(), math.tau)
            self.assertIsInstance(t.get_string(), str)
            self.assertEqual(t.get_string(), 'Godot')

        def test_math_types(self):
            t = GDExtensionTestCase()
            t.math_args_1([2.5, 5.], [5, 10], [0., 1., 100., 200.], [0, 1, 100, 200],
                          [2.5, 5., 10.], [5, 10, 20], [[1., 2.], [3., 4.], [5., 6.]],
                          [2.5, 5., 10., 20.], [5, 10, 20, 40])
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

            self.assertIsInstance(args[0], types.Vector2)
            self.assertEqual(args[0].dtype, np.dtype('float32'))
            self.assertEqual(args[0].tolist(), [2.5, 5.])

            self.assertIsInstance(args[1], types.Vector2i)
            self.assertEqual(args[1].dtype, np.dtype('int32'))
            self.assertEqual(args[1].tolist(), [5, 10])

            self.assertIsInstance(args[2], types.Rect2)
            self.assertEqual(args[2].dtype, np.dtype('float32'))
            self.assertEqual(args[2].tolist(), [0., 1., 100., 200.])

            self.assertIsInstance(args[3], types.Rect2i)
            self.assertEqual(args[3].dtype, np.dtype('int32'))
            self.assertEqual(args[3].tolist(), [0, 1, 100, 200])

            self.assertIsInstance(args[4], types.Vector3)
            self.assertEqual(args[4].dtype, np.dtype('float32'))
            self.assertEqual(args[4].tolist(), [2.5, 5., 10.])

            self.assertIsInstance(args[5], types.Vector3i)
            self.assertEqual(args[5].dtype, np.dtype('int32'))
            self.assertEqual(args[5].tolist(), [5, 10, 20])

            self.assertIsInstance(args[6], types.Transform2D)
            self.assertEqual(args[6].dtype, np.dtype('float32'))
            self.assertEqual(args[6].tolist(), [[1., 2.], [3., 4.], [5., 6.]])

            self.assertIsInstance(args[7], types.Vector4)
            self.assertEqual(args[7].dtype, np.dtype('float32'))
            self.assertEqual(args[7].tolist(), [2.5, 5., 10., 20.])

            self.assertIsInstance(args[8], types.Vector4i)
            self.assertEqual(args[8].dtype, np.dtype('int32'))
            self.assertEqual(args[8].tolist(), [5, 10, 20, 40])


class TestCaseArgTypes(BaseTestCase):
    def test_atomic_types(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        gdscript.call('test_atomic_types_out')

        r = gdscript.call('get_resource')

        self.assertIsInstance(r.arg1, bool)
        self.assertEqual(r.arg1, True)
        self.assertIsInstance(r.arg2, int)
        self.assertEqual(r.arg2, 42)
        self.assertIsInstance(r.arg3, float)
        self.assertEqual(r.arg3, math.tau)
        self.assertIsInstance(r.arg4, str)
        self.assertEqual(r.arg4, 'GodoPy')

        # TODO: Fix, it crashes on the Engine side: GDScriptDataType::is_type/Variant::get_type
        # gdscript.call('test_atomic_types_in_test', True)
        # a = gdscript.get('m_bool')
        # self.assertIsInstance(a, (bool, np.bool_))
        # self.assertEqual(a, True)


    def test_math_types(self):
        gdscript = self._main.get_node('TestCasesGDScript')

        for fn in ('test_math_types_1_out_1', 'test_math_types_1_out_2'):
            gdscript.call(fn)

            r = gdscript.call('get_resource')

            self.assertIsInstance(r.arg1, types.Vector2)
            self.assertEqual(r.arg1.dtype, np.dtype('float32'))
            self.assertEqual(r.arg1.tolist(), [2.5, 5.])

            self.assertIsInstance(r.arg2, types.Vector2i)
            self.assertEqual(r.arg2.dtype, np.dtype('int32'))
            self.assertEqual(r.arg2.tolist(), [5, 10])

            self.assertIsInstance(r.arg3, types.Rect2)
            self.assertEqual(r.arg3.dtype, np.dtype('float32'))
            self.assertEqual(r.arg3.tolist(), [0., 1., 100., 200.])

            self.assertIsInstance(r.arg4, types.Rect2i)
            self.assertEqual(r.arg4.dtype, np.dtype('int32'))
            self.assertEqual(r.arg4.tolist(), [0, 1, 100, 200])

            self.assertIsInstance(r.arg5, types.Vector3)
            self.assertEqual(r.arg5.dtype, np.dtype('float32'))
            self.assertEqual(r.arg5.tolist(), [2.5, 5., 10.])

            self.assertIsInstance(r.arg6, types.Vector3i)
            self.assertEqual(r.arg6.dtype, np.dtype('int32'))
            self.assertEqual(r.arg6.tolist(), [5, 10, 20])

            self.assertIsInstance(r.arg7, types.Transform2D)
            self.assertEqual(r.arg7.dtype, np.dtype('float32'))
            self.assertEqual(r.arg7.tolist(), [[1., 2.], [3., 4.], [5., 6.]])

            self.assertIsInstance(r.arg8, types.Vector4)
            self.assertEqual(r.arg8.dtype, np.dtype('float32'))
            self.assertEqual(r.arg8.tolist(), [2.5, 5., 10., 20.])

            self.assertIsInstance(r.arg9, types.Vector4i)
            self.assertEqual(r.arg9.dtype, np.dtype('int32'))
            self.assertEqual(r.arg9.tolist(), [5, 10, 20, 40])

        gdscript.call('test_math_types_2_out')
        r = gdscript.call('get_resource')

        self.assertIsInstance(r.arg1, types.Plane)
        self.assertEqual(r.arg1.dtype, np.dtype('float32'))
        self.assertEqual(r.arg1.tolist(), [5, 10, 20, 40])

        self.assertIsInstance(r.arg2, types.Quaternion)
        self.assertEqual(r.arg2.dtype, np.dtype('float32'))
        self.assertEqual(r.arg2.tolist(), [5, 10, 20, 40])

        self.assertIsInstance(r.arg3, types.AABB)
        self.assertEqual(r.arg3.dtype, np.dtype('float32'))
        self.assertEqual(r.arg3.tolist(), [[5, 10, 20], [40, 50, 60]])

        self.assertIsInstance(r.arg4, types.Basis)
        self.assertEqual(r.arg4.dtype, np.dtype('float32'))
        self.assertEqual(r.arg4.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        self.assertIsInstance(r.arg5, types.Transform3D)
        self.assertEqual(r.arg5.dtype, np.dtype('float32'))
        self.assertEqual(r.arg5.tolist(), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        self.assertIsInstance(r.arg6, types.Projection)
        self.assertEqual(r.arg6.dtype, np.dtype('float32'))
        self.assertEqual(r.arg6.tolist(), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

        self.assertIsInstance(r.arg7, types.Color)
        self.assertEqual(r.arg7.dtype, np.dtype('float32'))
        # Without converting to float32 numbers would differ due to the lost precision
        self.assertEqual(r.arg7.tolist(), [np.float32(n) for n in (0.2, 0.4, 0.5, 0.75)])


    def test_misc_types(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        gdscript.call('test_misc_types_out')
        r = gdscript.call('get_resource')

        self.assertIsInstance(r.arg1, types.StringName)
        self.assertEqual(r.arg1, "GodoPy")

        self.assertIsInstance(r.arg2, types.NodePath)
        self.assertEqual(str(r.arg2), "test/resource.py")

        self.assertIsInstance(r.arg3, types.RID)
        self.assertEqual(int(r.arg3), int(r.get_rid()))

        self.assertIsInstance(r.arg4, types.Object)
        self.assertEqual(r.arg4, r)

        # TODO: Callables and Signals

        self.assertIsInstance(r.arg7, dict)
        self.assertEqual(r.arg7, {"hello": 1, "world": 2.0, "!": ["!!!", [4, 5]]})

        self.assertIsInstance(r.arg8, list)
        self.assertEqual(r.arg8, [5, "a", 3.0, 40])

        # TODO: Typed arrays


    def test_packed_array_types(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        gdscript.call('test_packed_array_types_out')
        r = gdscript.call('get_resource')

        self.assertIsInstance(r.arg1, types.PackedByteArray)
        self.assertEqual(r.arg1.dtype, np.dtype('uint8'))
        self.assertEqual(r.arg1.tolist(), [2, 3, 4, 5, 1, 7])

        self.assertIsInstance(r.arg2, types.PackedInt32Array)
        self.assertEqual(r.arg2.dtype, np.dtype('int32'))
        self.assertEqual(r.arg2.tolist(), [2, 3, 4, 5, 1, 7])

        self.assertIsInstance(r.arg3, types.PackedInt64Array)
        self.assertEqual(r.arg3.dtype, np.dtype('int64'))
        self.assertEqual(r.arg3.tolist(), [2, 3, 4, 5, 1, 7])

        self.assertIsInstance(r.arg4, types.PackedFloat32Array)
        self.assertEqual(r.arg4.dtype, np.dtype('float32'))
        self.assertEqual([np.float32(x) for x in r.arg4.tolist()], [2., 3., 4., 5., 1.7, 7.])

        self.assertIsInstance(r.arg5, types.PackedFloat64Array)
        self.assertEqual(r.arg5.dtype, np.dtype('float64'))
        self.assertEqual(r.arg5.tolist(), [2., 3., 4.2, 5., 1., 7.])

        self.assertIsInstance(r.arg6, list)
        self.assertEqual(r.arg6, ['Hello', 'World', '!'])

        self.assertIsInstance(r.arg7, types.PackedVector2Array)
        self.assertEqual(r.arg7.dtype, np.dtype('float32'))
        self.assertEqual(r.arg7.shape[1], 2)
        self.assertIsInstance(r.arg7[0], types.Vector2)
        self.assertEqual(r.arg7.tolist(), [[2, 3], [4, 5], [1, 7]])

        self.assertIsInstance(r.arg8, types.PackedVector3Array)
        self.assertEqual(r.arg8.dtype, np.dtype('float32'))
        self.assertEqual(r.arg8.shape[1], 3)
        self.assertIsInstance(r.arg8[0], types.Vector3)
        self.assertEqual(r.arg8.tolist(), [[2, 3, 4], [9, 5, 6], [1, 7, 8]])

        self.assertIsInstance(r.arg9, types.PackedColorArray)
        self.assertEqual(r.arg9.dtype, np.dtype('float32'))
        self.assertEqual(r.arg9.shape[1], 4)
        self.assertIsInstance(r.arg9[0], types.Color)
        self.assertEqual(
            [[np.float32(y) for y in x] for x in r.arg9.tolist()],
            [[.2, .3, .4, 1.], [.9, .5, .6, 0.75], [.1, .7, .8, .1]]
        )

        self.assertIsInstance(r.argA, types.PackedVector4Array)
        self.assertEqual(r.argA.dtype, np.dtype('float32'))
        self.assertEqual(r.argA.shape[1], 4)
        self.assertIsInstance(r.argA[0], types.Vector4)
        self.assertEqual(r.argA.tolist(), [[2, 3, 4, 10], [9, 5, 6, 11], [1, 7, 8, 12]])
