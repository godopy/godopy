import importlib

import godot
from godot import classdb


class PythonRuntime(godot.Class, inherits=classdb.Object):
    @classdb.bind_method
    def import_module(self, name: str) -> object:
        return importlib.import_module(name)
