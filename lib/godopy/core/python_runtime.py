import importlib
import godot as gd

PythonRuntime = gd.ExtensionClass('PythonRuntime', 'Object')

@PythonRuntime.bind_method
def import_module(self, name: str) -> object:
    return importlib.import_module(name)
