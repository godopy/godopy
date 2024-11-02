import godot as gd
from godot import singletons as gds

from .core.python_runtime import PythonRuntime
from .core.python_script import (
    PythonScript,
    PythonLanguage,
    ResourceFormatLoaderPythonScript,
    ResourceFormatSaverPythonScript
)

python_runtime_singleton = None
python_language = None
loader = None
saver = None


def initialize(level):
    global python_runtime_singleton, python_language, loader, saver

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        PythonRuntime.register()
        python_runtime_singleton = PythonRuntime()
        gds.Engine.register_singleton('PythonRuntime', python_runtime_singleton)

        PythonLanguage.register()
        python_language = PythonLanguage()

        result = gds.Engine.register_script_language(python_language)
        if result != gd.Error.OK:
            raise RuntimeError("Could not register PythonScript language, error: %r" % gd.Error(result))

        PythonScript.register()
        ResourceFormatLoaderPythonScript.register()
        ResourceFormatSaverPythonScript.register()

        loader = ResourceFormatLoaderPythonScript()
        loader.reference()
        saver = ResourceFormatSaverPythonScript()
        saver.reference()
        gds.ResourceLoader.add_resource_format_loader(loader, True)
        gds.ResourceSaver.add_resource_format_saver(saver, True)


def uninitialize(level):
    global python_runtime_singleton, python_language, loader, saver

    if level == gd.MODULE_INITIALIZATION_LEVEL_SCENE:
        saver.unreference()
        gds.ResourceSaver.remove_resource_format_saver(saver)
        if saver is not None:
            saver.destroy()
            saver = None

        loader.unreference()
        gds.ResourceLoader.remove_resource_format_loader(loader)
        if loader is not None:
            loader.destroy()
            loader = None

        gds.Engine.unregister_script_language(python_language)
        python_language.destroy()
        python_language = None

        gds.Engine.unregister_singleton('PythonRuntime')
        python_runtime_singleton.destroy()
        python_runtime_singleton = None
