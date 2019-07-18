import inspect
from collections import OrderedDict

from Cython.Shadow import *

class Registry:
    def __init__(self):
        self.cimports = OrderedDict()
        self.classes = []

registry = Registry()

def cimport(module, names):
    registry.cimports.setdefault(module, set()).update(names)
    return tuple(lambda *args, **kwargs: name for name in names)

class register_class:
    def __init__(self, name, base):
        self.name = name
        self.base = base

        self.attrs = {}
        self.methods = {}

        self.props = {}
        self.getset_props = {}

        self.public_methods = []

        self.signals = {}

        registry.classes.append(self)
        cimport('godot_cpp.gen', [base])

    def attr(self, attr, type):
        self.attrs[attr] = type

    def method(self, func):
        spec = inspect.getfullargspec(func)
        spec.annotations['self'] = f'{self.name} *'
        if 'return' not in spec.annotations:
            spec.annotations['return'] = void
        self.methods[func.__name__] = spec
        return func

    def register_property(self, prop, *args):
        if len(args) == 2:
            type, value = args
            self.attr(prop, type)
            self.props[prop] = dict(type=type, value=value)
        elif len(args) == 3:
            setter, getter, value = args
            self.getset_props[prop] = dict(setter=setter.__name__, getter=getter.__name__, value=value, type=self.attrs[prop])
        else:
            raise ArgumentError(f'Invalid number of arguments for "{self.name}.{prop}"')

    def register_method(self, func):
        self.public_methods.append(func.__name__)
        return self.method(func)

    def register_signal(self, signal, **kwargs):
        self.signals[signal] = kwargs
