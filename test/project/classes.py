import math
import numpy as np

import godot as gd
from godot import classdb, types


class Example(gd.Class, inherits=classdb.Control):
    def __init__(self):
        self.entered_tree = False
        self.exited_tree = False

    def _enter_tree(self):
        self.entered_tree = True

    def _exit_tree(self):
        self.exited_tree = True


class TestResource(gd.Class, inherits=classdb.Resource):
    def __init__(self):
        self.arg01 = None
        self.arg02 = None
        self.arg03 = None
        self.arg04 = None

    @gd.method
    def atomic_args(self, arg01: bool, arg02: int, arg03: float, arg04: str):
        self.arg01 = arg01
        self.arg02 = arg02
        self.arg03 = arg03
        self.arg04 = arg04

    @gd.method
    def bool_ret(self) -> bool:
        return True
    
    @gd.method
    def int_ret(self) -> int:
        return 42

    @gd.method
    def float_ret(self) -> float:
        return math.tau

    @gd.method
    def string_ret(self) -> str:
        return 'GodoPy'

    @gd.method
    def math_args_1(self, arg01: types.Vector2, arg02: types.Vector2i, arg03: types.Rect2, arg04: types.Rect2i):
        self.arg01 = arg01
        self.arg02 = arg02
        self.arg03 = arg03
        self.arg04 = arg04
