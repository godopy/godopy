import importlib

import godot as gd
from godot import classdb


class PythonRuntime(classdb.Object):
    @gd.method
    def import_module(self, name: str) -> object:
        return importlib.import_module(name)
