"""
GodoPy testing framework

Run with::
 
    <godot-executable> --path test/project --headless --quit

or::

    <godot-executable> --path test/project --headless --verbose-tests --quit
"""
import math
import unittest

import godot as gd
import gdextension as gde
from godot import classdb

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

    def test_amath_types_1(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        mb = gde.MethodBind(gdscript, 'call')
        mb.call('test_math_types_1')

        r = mb.call('get_resource')

        self.assertIsInstance(r.arg01, tuple)
        self.assertIsInstance(r.arg01[0], float)
        self.assertEqual(r.arg01, (2.5, 5.0))
        self.assertIsInstance(r.arg02, tuple)
        self.assertIsInstance(r.arg02[0], int)
        self.assertEqual(r.arg02, (5, 10))
        self.assertIsInstance(r.arg03, tuple)
        self.assertIsInstance(r.arg03[0], tuple)
        self.assertIsInstance(r.arg03[0][0], float)
        self.assertEqual(r.arg03, ((0.0, 0.0), (100.0, 200.0)))
        self.assertIsInstance(r.arg04, tuple)
        self.assertIsInstance(r.arg04[0], tuple)
        self.assertIsInstance(r.arg04[0][0], int)
        self.assertEqual(r.arg03, ((0, 0.0), (100, 200)))

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
