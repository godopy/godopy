import numpy as np

import godot as gd
from godot import classdb


class ExampleRef(gd.Class, inherits=classdb.RefCounted):
    instance_count = 0
    last_id = 0

    def __init__(self):
        self.id = None
        self.post_initialized = False

    def set_id(self, id: int):
        self.id = id

    def get_id(self) -> int:
        return self.id

    def was_post_initialized(self):
        return self.post_initialized


class ExampleMin(gd.Class, inherits=classdb.Control):
    pass


class Example(gd.Class, inherits=classdb.Control):
    pass
