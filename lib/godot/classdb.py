import godot as gd
import gdextension as gde

__all__ = list(gde._classdb_dir())


def __getattr__(name):
    if gde._has_class(name):
        cls = gd.GodotClassBase(name, (gd.EngineClass,), {'__godot_class__': gde.Class._get_class(name)})
        globals()[name] = cls

        return cls

    # TODO: Extension classes

    else:
        raise AttributeError('%r class does not exist' % name)


def __dir__():
    return __all__
