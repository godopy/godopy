"""
Contains all Godot singletons that are available classes in the Extension API.
All singleton objects in this module are loaded only when they are needed.
"""
from typing import List, Any

import godot
import gdextension


__all__ = list(gdextension.singletons_set())


__path__ = None


def __getattr__(name) -> gdextension.Object:
    """
    Creates and returns an instance of godot.EngineClass for all available singletons in the Extension API.
    """
    try:
        if gdextension.has_singleton(name):
            godot_class = gdextension.Class.get(name)
            cls = godot.GodotClassBase(name, (godot.EngineClass,), {'__godot_class__': godot_class})

            singleton = cls()
            globals()[name] = singleton

            return singleton

        else:
            raise AttributeError('%r singleton does not exist' % name)
    except Exception as exc:
        # Make errors visible when imported directly as `from godot.singletons import <Singleton>``
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
    Lists all available singletons.
    """
    return __all__
