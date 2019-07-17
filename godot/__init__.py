import os
import sys
from .version import get_version

inside_godot = False

_rootdir = os.path.abspath((os.path.join(os.path.dirname(__file__), '..', '..', '..')))

if (os.path.isdir(_rootdir) and _rootdir.endswith('pygodot.resources')):
    inside_godot = True

if inside_godot:
    from utils import _pyprint as print
else:
    class dummy_nodes:
        def __init__(self):
            self.Reference = type('Reference', (object,), {})
    class dummy_gdnative:
        def register_class(self, cls): pass
        def register_method(self, cls, method): pass

    nodes = dummy_nodes()
    gdnative = dummy_gdnative()
    print = print

__version__ = get_version()
