import gdextension as gde


__all__ = list(gde._singletons_dir())


class Singleton(gde.Object):
    def __getattr__(self, name):
        mb = gde.MethodBind(self, name)

        self.__dict__[name] = mb

        return mb


    def __dir__(self):
        return self.__godot_class__.__method_info__.keys()


def __getattr__(name):
    if gde._has_singleton(name):
        return Singleton(name)
    # TODO: Extension singletons
    else:
        raise AttributeError('%r singleton does not exist' % name)


def __dir__():
    return __all__
