import godot as gd

python_runtime_singleton = None


def initialize(level):
    global python_runtime_singleton

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        from godot.singletons import Engine
        from .core.python_runtime import PythonRuntime

        PythonRuntime.register()
        python_runtime_singleton = PythonRuntime()
        Engine.register_singleton('PythonRuntime', python_runtime_singleton)


def deinitialize(level):
    global python_runtime_singleton

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        from godot.singletons import Engine

        Engine.unregister_singleton('PythonRuntime')
        python_runtime_singleton.destroy()
        python_runtime_singleton = None
