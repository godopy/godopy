from gdextension import Editor

import godot
from godot.classdb import Engine

from .core.python_runtime import PythonRuntime
from .scene import PythonScenePlugin

python_runtime_singleton = None


def initialize(level: int) -> None:
    global python_runtime_singleton

    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        PythonRuntime.register()
        python_runtime_singleton = PythonRuntime()
        Engine.register_singleton('PythonRuntime', python_runtime_singleton)

    elif level == godot.MODULE_INITIALIZATION_LEVEL_EDITOR:
        PythonScenePlugin.register_internal()
        Editor.add_plugin('PythonScenePlugin')


def uninitialize(level: int) -> None:
    global python_runtime_singleton

    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        Engine.unregister_singleton('PythonRuntime')
        python_runtime_singleton.destroy()
        python_runtime_singleton = None
