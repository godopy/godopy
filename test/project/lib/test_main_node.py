import unittest

import godot as gd
from godot import classdb

from example import Example


class TestCase(unittest.TestCase):
    def test_upper(self):
        example = Example()
        self.assertGreater(example.owner_id(), 0)


class TestMainNode(gd.Class, inherits=classdb.Node):
    def __init__(self):
        self.test_passes = 0
        self.test_failures = 0

        self.suite = unittest.TestSuite()

        for attr, TestCase in globals().items():
            if attr.startswith('TestCase'):
                for attr in dir(TestCase):
                    if attr.startswith('test_'):
                        self.suite.addTest(TestCase(attr))

    def _ready(self):
        verbosity = 2 if gd.singletons.OS.is_stdout_verbose() else 1
        if '--verbose-tests' in gd.singletons.OS.get_cmdline_args():
            verbosity = 2
        runner = unittest.TextTestRunner(verbosity=verbosity)
        runner.run(self.suite)
