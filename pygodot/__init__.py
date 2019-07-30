from .version import get_version

try:
    __import__('_python_bindings')
except ImportError:
    inside_godot = False
else:
    inside_godot = True

__version__ = get_version()
