from Cython.Shadow import *

def cimport(module, symbols):
    return tuple(lambda *args, **kwargs: s for s in symbols)

class register_class:
    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.attrs = {}
        self.props = {}
        self.methods = {}

    def declare_attr(self, attr, type):
        self.attrs[attr] = type

    def declare_prop(self, prop, type):
        self.props[prop] = type

    def register_method(self, func):
        self.methods[func.__name__] = func
        return func
