import unittest

from godot import bindings
from godot.nativescript import register_tool_class, register_method

import numpy as np
from godot.core.types import Vector2


class TestVector2Methods(unittest.TestCase):
    def test_to_numpy(self):
        self.assertTrue(np.array_equal(Vector2(2, 5).to_numpy(), np.array([2, 5], dtype=np.float32)))


class Main(bindings.MainLoop):
    def _initialize(self):
        self.suite = unittest.TestSuite()
        self.suite.addTest(TestVector2Methods('test_to_numpy'))

        self.runner = unittest.TextTestRunner()

    def _idle(self, delta):
        self.runner.run(self.suite)

        return True

    @staticmethod
    def _register_methods():
        register_method(Main, '_initialize')
        register_method(Main, '_idle')


def _init():
    register_tool_class(Main)
