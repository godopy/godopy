import sys
import godot as gd

from example import Example
from test_main import TestMain

def initialize(level):
    if level != 2:
        return

    Example.register()
    TestMain.register()
