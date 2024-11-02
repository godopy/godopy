import godot
from godot.singletons import Engine

from .core.python_runtime import PythonRuntime


python_runtime_singleton = None


def initialize(level):
    global python_runtime_singleton

    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        PythonRuntime.register()
        python_runtime_singleton = PythonRuntime()
        Engine.register_singleton('PythonRuntime', python_runtime_singleton)


def uninitialize(level):
    global python_runtime_singleton

    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        Engine.unregister_singleton('PythonRuntime')
        python_runtime_singleton.destroy()
        python_runtime_singleton = None
