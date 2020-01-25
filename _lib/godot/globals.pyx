from .core cimport cpp_types as cpp


def gdprint(fmt, *args):
    fmt = str(fmt)

    if not args:
        Godot.print(fmt)
    elif len(args) == 1:
        Godot.print(fmt, <cpp.Variant>args[0])
    elif len(args) == 2:
        Godot.print(fmt, <cpp.Variant>args[0], <cpp.Variant>args[1])
    elif len(args) == 3:
        Godot.print(fmt, <cpp.Variant>args[0], <cpp.Variant>args[1], <cpp.Variant>args[2])
    elif len(args) == 4:
        Godot.print(fmt, <cpp.Variant>args[0], <cpp.Variant>args[1], <cpp.Variant>args[2], <cpp.Variant>args[3])
    else:
        Godot.print(fmt, <cpp.Variant>args[0], <cpp.Variant>args[1], <cpp.Variant>args[2], <cpp.Variant>args[3], <cpp.Variant>args[4])


def gdclass(cls=None, single_method=None, register_all=False, _force_decorator=False):
    if single_method is not None:
        return _convert_to_class(cls, single_method)
    elif isinstance(cls, type) and '__init__' in cls.__dict__ and not _force_decorator:
        # XXX: '__init__' in cls.__dict__ is a hacky way to differenciate between `@gdclass(Base)` and `@gdclass`
        base = cls

        def _gdclass(cls):
            real_cls = type(cls.__name__, (base,), dict(cls.__dict__))
            return gdclass(real_cls, _force_decorator=True)

        return _gdclass

    cls.__godot_methods__ = []
    cls.__godot_properties__ = {}
    cls.__godot_signals__ = {}

    for name, attr in cls.__dict__.items():
        if hasattr(attr, '__add_to_godot_methods') or (register_all and not hasattr(attr, '__exclude_from_godot_methods') and callable(attr)):
            cls.__godot_methods__.append(name)
        if hasattr(attr, '__add_to_godot_properties'):
            cls.__godot_properties__[name] = attr.__default_value
        if hasattr(attr, '__add_to_godot_signals'):
            cls.__godot_signals__[name] = attr.__signal_args

    return cls


def _convert_to_class(base, single_method):
    def _gdclass_from_function(func):
        return type(func.__name__, (base,), {
            single_method: func,
            '__godot_methods__': [single_method]
        })

    return _gdclass_from_function
