import gdextension as gde
import godot as gd

__all__ = list(gde._classdb_dir())


def __getattr__(name):
    if gde._has_class(name):
        return gd.GodotClassBase(name, (gd.EngineClass,), {'__godot_class__': gde.Class._get_class(name)})
    # TODO: Extension classes
    else:
        raise AttributeError('%r class does not exist' % name)


def __dir__():
    return __all__
