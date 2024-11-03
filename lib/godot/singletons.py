import godot as gd
import gdextension as gde


__all__ = list(gde._singletons_dir())


__path__ = None


def __getattr__(name):
    try:
        if gde._has_singleton(name):
            godot_class = gde.Class.get(name)
            cls = gd.GodotClassBase(name, (gd.EngineClass,), {'__godot_class__': godot_class})

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


def __dir__():
    return __all__
