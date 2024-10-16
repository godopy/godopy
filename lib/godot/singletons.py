import godot as gd
import gdextension as gde


__all__ = list(gde._singletons_dir())


def __getattr__(name):
    if gde._has_singleton(name):
        cls = gd.GodotClassBase(name, (gd.EngineClass,), {'__godot_class__': gde.Class._get_class(name)})
        singleton = cls()
        globals()[name] = singleton

        return singleton

    # TODO: Extension singletons

    else:
        raise AttributeError('%r singleton does not exist' % name)


def __dir__():
    return __all__
