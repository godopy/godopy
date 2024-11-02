import godot as gd
import gdextension as gde

__all__ = ['bind_method', 'bind_vurtual_method'] + list(gde._classdb_dir())


def bind_method(func):
    func._gdmethod = func.__name__
    return func


def bind_virtual_method(func):
    func._gdvirtualmethod = func.__name__
    return func


def __getattr__(name):
    if gde._has_class(name):
        cls = gd.GodotClassBase(name, (gd.EngineClass,), {'__godot_class__': gde.Class._get_class(name)})
        globals()[name] = cls

        return cls
    else:
        raise AttributeError('%r class does not exist' % name)


def __dir__():
    return __all__
