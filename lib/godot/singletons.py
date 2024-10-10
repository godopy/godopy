import gdextension as gd


__all__ = list(gd._singletons_dir())


class Singleton(gd.Object):
    def __getattr__(self, name):
        mb = gd.MethodBind(self, name)

        self.__dict__[name] = mb

        return mb


    def __dir__(self):
        return self.__godot_class__.__method_info__.keys()


def __getattr__(name):
    if gd._has_singleton(name):
        return Singleton(name)
    else:
        raise AttributeError('%r singleton does not exist' % name)


def __dir__():
    return __all__
