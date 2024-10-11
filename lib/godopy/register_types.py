import godot as gd

from .core.python_runtime import PythonRuntime
from .core.python_script import ResourceFormatLoaderPythonScript

python_runtime_singleton = None
loader = None


def initialize(level):
    global python_runtime_singleton, loader

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        from godot.singletons import Engine, ResourceLoader

        PythonRuntime.register()
        python_runtime_singleton = PythonRuntime()
        Engine.register_singleton('PythonRuntime', python_runtime_singleton)

        # PythonScript.register()
        ResourceFormatLoaderPythonScript.register()
        loader = ResourceFormatLoaderPythonScript()
        loader.reference()
        ResourceLoader.add_resource_format_loader(loader, False)


def deinitialize(level):
    global python_runtime_singleton, loader

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        from godot.singletons import Engine, ResourceLoader

        ResourceLoader.remove_resource_format_loader(loader)
        loader.unreference()
        loader.destroy()
        loader = None

        Engine.unregister_singleton('PythonRuntime')
        python_runtime_singleton.destroy()
        python_runtime_singleton = None
