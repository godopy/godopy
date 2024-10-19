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


class TestCaseArgTypes(BaseTestCase):
    def test_atomic_types(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        mb = gde.MethodBind(gdscript, 'call')
        mb.call('test_atomic_types')

        r = mb.call('get_resource')

        self.assertIsInstance(r.arg01, bool)
        self.assertEqual(r.arg01, True)
        self.assertIsInstance(r.arg02, int)
        self.assertEqual(r.arg02, 42)
        self.assertIsInstance(r.arg03, float)
        self.assertEqual(r.arg03, math.tau)
        self.assertIsInstance(r.arg04, str)
        self.assertEqual(r.arg04, 'GodoPy')

    def test_math_types_1(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        mb = gde.MethodBind(gdscript, 'call')
        mb.call('test_math_types_1')

        r = mb.call('get_resource')

        self.assertIsInstance(r.arg01, types.Vector2)
        self.assertEqual(r.arg01.dtype, np.dtype('float32'))
        self.assertIsInstance(r.arg01[0], np.float32)
        self.assertIsInstance(r.arg01.x, np.float32)
        self.assertEqual(list(r.arg01), [2.5, 5.])

        r.arg01.x = 10.0
        self.assertEqual(list(r.arg01), [10., 5.])

        r.arg01.coord = (2, 1)
        self.assertEqual(list(r.arg01), [2., 1.])

        self.assertIsInstance(r.arg02, types.Vector2i)
        self.assertEqual(r.arg02.dtype, np.dtype('int32'))
        self.assertIsInstance(r.arg02.x, np.int32)
        self.assertEqual(list(r.arg02), [5, 10])

        self.assertIsInstance(r.arg03, types.Rect2)
        self.assertIsInstance(r.arg03.position, types.Vector2)
        self.assertIsInstance(r.arg03.position.x, np.float32)
        self.assertIsInstance(r.arg03.size, types.Size2)
        self.assertEqual(list(r.arg03), [0., 0., 100., 200.])

        r.args03.position = (2, 5)
        self.assertEqual(list(r.arg03), [2., 5., 100., 200.])

        r.args03.position.x = 10
        self.assertEqual(list(r.arg03), [10., 5., 100., 200.])

        r.args03.size.height = 50
        self.assertEqual(list(r.arg03), [10., 5., 100., 50.])

        self.assertEqual(list(r.arg03.position), [10., 5.])
        self.assertEqual(list(r.arg03.position), [100., 50.])

        self.assertIsInstance(r.arg04, types.Rect2i)
        self.assertIsInstance(r.arg04.position, types.Vector2i)
        self.assertIsInstance(r.arg04.position.x, np.int32)
        self.assertEqual(list(r.arg04), [0, 0, 100, 200])

class TestCaseSceneExtension(BaseTestCase):
    def test_owner(self):
        example = self._main.get_node('Example')
        self.assertGreater(example.owner_id(), 0)

    def test_class(self):
        example = self._main.get_node('Example')
        self.assertEqual(example.__godot_class__.__name__, 'Example')
        self.assertEqual(example.__class__.__godot_class__.__get__(example), example.__godot_class__)


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
