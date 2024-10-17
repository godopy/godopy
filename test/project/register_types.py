import sys
import godot as gd

from classes import Example, TestResource
from test_main import TestMain


def initialize(level):
    if level != 2:
        return

    Example.register()
    TestResource.register()
    TestMain.register()
