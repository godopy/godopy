import godot as gd
from godot import singletons as gds

from .core.python_runtime import PythonRuntime
from .core.python_script import PythonScript, ResourceFormatLoaderPythonScript

python_runtime_singleton = None
loader = None


def initialize(level):
    global python_runtime_singleton, loader

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        PythonRuntime.register()
        python_runtime_singleton = PythonRuntime()
        gds.Engine.register_singleton('PythonRuntime', python_runtime_singleton)

        PythonScript.register()
        ResourceFormatLoaderPythonScript.register()
        loader = ResourceFormatLoaderPythonScript()
        loader.reference()
        gds.ResourceLoader.add_resource_format_loader(loader, True)


def deinitialize(level):
    global python_runtime_singleton, loader

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        gds.ResourceLoader.remove_resource_format_loader(loader)
        loader.unreference()
        loader.destroy()
        loader = None

        gds.Engine.unregister_singleton('PythonRuntime')
        python_runtime_singleton.destroy()
        python_runtime_singleton = None
