"""
Contains all Godot classes that are available classes in the Extension API.
All classes in this module are loaded only when they are needed.
"""
from typing import Callable, List

import godot
import gdextension


__all__ = ['bind_method', 'bind_vurtual_method'] + list(gdextension.classdb_set())


__path__ = None


def bind_method(func: Callable) -> Callable:
    """
    Decorates a method to make it available as a method of the custom Godot class.
    """
    func._gdmethod = func.__name__

    return func


def bind_virtual_method(func: Callable) -> Callable:
    """
    Decorates a method to make it available as a virtual method of the custom Godot class.
    """
    func._gdvirtualmethod = func.__name__

    return func


def __getattr__(name) -> type:
    """
    Creates and returns a subclass of godot.EngineClass for all available classes in the Extension API.
    """
    try:
        if gdextension.has_class(name):
            godot_class = gdextension.Class.get(name)
            cls = godot.GodotClassBase(name, (godot.EngineClass,), {'__godot_class__': godot_class})

            globals()[name] = cls

            return cls

        else:
            raise AttributeError('%r class does not exist' % name)
    except Exception as exc:
        # Make errors visible when imported directly as `from godot.classdb import <Class>``
        import inspect
        import re

        frame = inspect.currentframe()
        context = inspect.getframeinfo(frame.f_back).code_context

        if context and re.search(r"from .* import", context[0]):
            raise ImportError(exc)
        else:
            raise exc


def __dir__() -> List[str]:
    """
    Lists all available classes.
    """
    return __all__
