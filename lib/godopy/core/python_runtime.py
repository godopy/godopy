import importlib
from typing import Any

import godot
from godot.classdb import Object, bind_method


class PythonRuntime(godot.Class, inherits=Object):
    @bind_method
    def import_module(self, name: str) -> Any:
        return importlib.import_module(name)
