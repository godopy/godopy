import sys
import godot as gd

from classes import Example, TestResource, ExtendedTestObject
from test_main import TestMain


def initialize(level):
    if level != 2:
        return

    Example.register()
    TestResource.register()

    if ExtendedTestObject is not None:
        ExtendedTestObject.register()

    TestMain.register()
