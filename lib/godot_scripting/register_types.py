import godot
from godot.singletons import Engine, ResourceLoader, ResourceSaver

from .script import Python
from .language import PythonLanguage
from .resource_format import ResourceFormatLoaderPython, ResourceFormatSaverPython

import godot.singletons


python_language = None
script_loader = None
script_saver = None


def initialize(level: int) -> None:
    global python_language, script_loader, script_saver

    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        PythonLanguage.register()
        python_language = PythonLanguage()

        result = Engine.register_script_language(python_language)
        if result != godot.Error.OK:
            raise RuntimeError("Could not register PythonScript language, error: %r" % godot.Error(result))

        Python.register()
        ResourceFormatLoaderPython.register()
        ResourceFormatSaverPython.register()

        script_loader = ResourceFormatLoaderPython()
        script_loader.reference()
        ResourceLoader.add_resource_format_loader(script_loader, True)

        script_saver = ResourceFormatSaverPython()
        script_saver.reference()
        ResourceSaver.add_resource_format_saver(script_saver, True)


def uninitialize(level: int) -> None:
    global script_saver, script_loader, python_language

    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        script_saver.unreference()
        ResourceSaver.remove_resource_format_saver(script_saver)
        script_saver.destroy()
        script_saver = None


        script_loader.unreference()
        ResourceLoader.remove_resource_format_loader(script_loader)
        script_loader.destroy()
        script_loader = None

        Engine.unregister_script_language(python_language)
        python_language.destroy()
        python_language = None