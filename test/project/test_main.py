"""
GodoPy testing framework

Run with::
 
    <godot-executable> --path test/project --headless --quit

or::

    <godot-executable> --path test/project --headless --verbose-tests --quit
"""
import numpy as np
import math
import unittest

import godot as gd
import gdextension as gde
from godot import classdb, types

from classes import TestResource


class BaseTestCase(unittest.TestCase):
    def __init__(self, main_obj, method_name):
        super().__init__(method_name)

        self._main = main_obj


class TestCaseEngineSingleton(BaseTestCase):
    def test_owner(self):
        Engine = gd.singletons.Engine
        self.assertGreater(Engine.owner_id(), 0)

    def test_str_str_method(self):
        ProjectSettings = gd.singletons.ProjectSettings
        self.assertEqual(ProjectSettings.get('application/run/main_scene'), 'res://main.tscn')


class TestCaseAtomicTypes(BaseTestCase):
    def test_string(self):
        s = types.String('GodoPy')
        self.assertIsInstance(s, str)
        self.assertIsInstance(s, types.String)

        self.assertEqual(s, 'GodoPy')

        s2 = s.to_snake_case()
        self.assertEqual(s2, 'godo_py')

        self.assertEqual(s2.to_camel_case(), 'godoPy')
        self.assertEqual(s2.to_pascal_case(), 'GodoPy')
        self.assertEqual(s2.capitalize(), 'Godo_py')  # native Python str's method
        self.assertEqual(s2.upper(), s2.to_upper())
        self.assertEqual(s2.lower(), s2.to_lower())
        self.assertEqual(s2.is_empty(), False)
        self.assertEqual(types.String().is_empty(), True)
        self.assertEqual(s.path_join('tests').path_join('string'), 'GodoPy/tests/string')

class TestCaseMathTypes(BaseTestCase):
    def test_vector2(self):
        v = types.Vector2(2.5, 5)
        self.assertEqual(v.dtype, np.dtype('float32'))
        self.assertIsInstance(v[0], np.float32)
        self.assertIsInstance(v.x, np.float32)
        self.assertEqual(list(v), [2.5, 5.])

        v.x = 10.0
        self.assertEqual(list(v), [10., 5.])

        v.coord = (2, 1)
        self.assertEqual(list(v), [2., 1.])

        v2 = types.Vector2i(5, 10)
        self.assertIsInstance(v2, types.Vector2i)
        self.assertEqual(v2.dtype, np.dtype('int32'))
        self.assertIsInstance(v2.x, np.int32)
        self.assertEqual(list(v2), [5, 10])

        v3 = types.as_vector2(v2)
        self.assertIsInstance(v3, types.Vector2)
        self.assertEqual(v3.dtype, np.dtype('float32'))
        self.assertEqual(list(v3), [5., 10.])

        v4 = types.as_vector2(v3, dtype=np.float64)
        self.assertIsInstance(v4, types.Vector2)
        self.assertEqual(v4.dtype, np.dtype('float64'))

        v5 = types.as_vector2i(v4, dtype=np.int8)
        self.assertIsInstance(v5, types.Vector2i)
        self.assertEqual(v5.dtype, np.dtype('int8'))

    def test_rect(self):
        r = types.Rect2(0, 0, 100, 200)
        self.assertIsInstance(r, types.Rect2)
        self.assertEqual(r.dtype, np.dtype('float32'))
        self.assertIsInstance(r.position, types.Vector2)
        self.assertIsInstance(r.position.x, np.float32)
        self.assertIsInstance(r.size_, types.Size2)
        self.assertEqual(list(r), [0., 0., 100., 200.])

        r.position = (2, 5)
        self.assertEqual(list(r), [2., 5., 100., 200.])

        r.position.x = 10
        self.assertEqual(list(r), [10., 5., 100., 200.])

        r.size_.height = 50
        self.assertEqual(list(r), [10., 5., 100., 50.])

        self.assertEqual(list(r.position), [10., 5.])
        self.assertEqual(list(r.size_), [100., 50.])

        r2 = types.Rect2i(0, 0, 100, 200)
        self.assertIsInstance(r2, types.Rect2i)
        self.assertEqual(r2.dtype, np.dtype('int32'))
        self.assertIsInstance(r2.position, types.Vector2i)
        self.assertIsInstance(r2.position.x, np.int32)
        self.assertEqual(list(r2), [0, 0, 100, 200])

        # Type casting and reshaping
        r3 = types.as_rect2(np.array([[10, 20], [100, 80]]))
        self.assertEqual(r.dtype, np.dtype('float32'))
        self.assertEqual(list(r3), [10, 20, 100, 80])

class TestCaseArgTypes(BaseTestCase):
    def test_atomic_types(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        gdscript.call('test_atomic_types')

        r = gdscript.call('get_resource')

        self.assertIsInstance(r.arg01, (bool, np.bool_))
        self.assertEqual(r.arg01, True)
        self.assertIsInstance(r.arg02, np.integer)
        self.assertEqual(r.arg02, 42)
        self.assertIsInstance(r.arg03, np.floating)
        self.assertEqual(r.arg03, math.tau)
        self.assertIsInstance(r.arg04, str)
        self.assertEqual(r.arg04, 'GodoPy')

    def test_math_types_1(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        gdscript.call('test_math_types_1')

        r = gdscript.call('get_resource')

        self.assertIsInstance(r.arg01, types.Vector2)
        self.assertEqual(r.arg01.dtype, np.dtype('float32'))
        self.assertEqual(list(r.arg01), [2.5, 5.])
        self.assertIsInstance(r.arg02, types.Vector2i)
        self.assertEqual(r.arg02.dtype, np.dtype('int32'))
        self.assertEqual(list(r.arg02), [5, 10])
        self.assertIsInstance(r.arg03, types.Rect2)
        self.assertEqual(r.arg03.dtype, np.dtype('float32'))
        self.assertEqual(list(r.arg03), [0., 0., 100., 200.])
        self.assertIsInstance(r.arg04, types.Rect2i)
        self.assertEqual(r.arg04.dtype, np.dtype('int32'))
        self.assertEqual(list(r.arg03), [0, 0, 100, 200])


class TestCaseSceneExtension(BaseTestCase):
    def test_owner(self):
        example = self._main.get_node('Example')
        self.assertGreater(example.owner_id(), 0)

    def test_class(self):
        example = self._main.get_node('Example')
        self.assertEqual(example.__godot_class__.__name__, 'Example')
        self.assertEqual(example.__class__.__godot_class__, example.__godot_class__)

        with self.assertRaises(TypeError) as ctx:
            example.__class__(kwarg=1)
        
        self.assertTrue("unexpected keyword argument 'kwarg'" in str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            example.__class__(_notify=False)


class TestStream:
    def __init__(self):
        self._test_data = []

    def write(self, data):
        gd.printraw(data)

    def flush(self):
        pass


class TestMain(gd.Class, inherits=classdb.Node):
    def __init__(self):
        self.suite = unittest.TestSuite()

        for attr_name, TestCase in globals().items():
            if attr_name.startswith('TestCase'):
                self.suite.addTests(
                    TestCase(self, cls_attr_name) for cls_attr_name in dir(TestCase)
                                                  if cls_attr_name.startswith('test_')
                )

    def _ready(self):
        verbosity = 2 if gd.singletons.OS.is_stdout_verbose() else 1
        if '--verbose-tests' in gd.singletons.OS.get_cmdline_args():
            verbosity = 2

        stream = TestStream()
        runner = unittest.TextTestRunner(stream, verbosity=verbosity)
        runner.run(self.suite)
