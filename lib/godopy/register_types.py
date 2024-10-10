import importlib
import godot as gd

python_runtime_singleton = None


def python_runtime_class():
    PythonRuntime = gd.ExtensionClass('PythonRuntime', 'Object')

    @PythonRuntime.bind_method
    def import_module(self, name: str) -> object:
        return importlib.import_module(name)

    return PythonRuntime


def initialize(level):
    global python_runtime_singleton

    if level == 2:
        PythonRuntime = python_runtime_class()
        PythonRuntime.register()

        Engine = gd.Object('Engine')
        register_singleton = gd.MethodBind(Engine, 'register_singleton')

        python_runtime_singleton = PythonRuntime()
        register_singleton('PythonRuntime', python_runtime_singleton)


def deinitialize(level):
    global python_runtime_singleton

    if level == 2:
        Engine = gd.Object('Engine')
        unregister_singleton = gd.MethodBind(Engine, 'unregister_singleton')
        unregister_singleton('PythonRuntime')

        python_runtime_singleton.destroy()

        python_runtime_singleton = None
