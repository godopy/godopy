import godot as gd
import gdextension as gde


__all__ = ['bind_method', 'bind_vurtual_method'] + list(gde._classdb_dir())


__path__ = None


def bind_method(func):
    func._gdmethod = func.__name__
    return func


def bind_virtual_method(func):
    func._gdvirtualmethod = func.__name__
    return func


def __getattr__(name):
    try:
        if gde._has_class(name):
            godot_class = gde.Class.get(name)
            cls = gd.GodotClassBase(name, (gd.EngineClass,), {'__godot_class__': godot_class})

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


def __dir__():
    return __all__
