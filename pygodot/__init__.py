from .version import get_version

try:
    __import__('cnodes')
    inside_godot = True
except ImportError:
    inside_godot = False

if inside_godot:
    from utils import _pyprint as print
else:
    class dummy_bindings:
        pass

    class dummy_gdnative:
        def register_class(self, cls): pass
        def register_method(self, cls, method): pass

    bindings = dummy_bindings()
    gdnative = dummy_gdnative()
    print = print

__version__ = get_version()
