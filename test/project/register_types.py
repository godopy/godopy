import sys
import godot as gd

from example import ExampleRef, ExampleMin, Example
from test_main_node import TestMainNode

def initialize(level):
    if level != 2:
        return

    ExampleRef.register()
    ExampleMin.register()
    Example.register()

    TestMainNode.register()
