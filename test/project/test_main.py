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

import tests
from tests._base import BaseTestCase
from classes import TestResource


class TestCaseEngineSingleton(BaseTestCase):
    def test_owner(self):
        Engine = gd.singletons.Engine
        self.assertGreater(Engine.owner_id(), 0)

    def test_str_str_method(self):
        ProjectSettings = gd.singletons.ProjectSettings
        self.assertEqual(ProjectSettings.get('application/run/main_scene'), 'res://main.tscn')

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
        for mod_name in dir(tests):
            mod = getattr(tests, mod_name)
            for case_name in dir(mod):
                TestCase = getattr(mod, case_name)
                if case_name.startswith('TestCase'):
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
