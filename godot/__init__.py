import os
import sys
from .version import get_version

inside_godot = False

_rootdir = os.path.abspath((os.path.join(os.path.dirname(__file__), '..', '..', '..')))

if (os.path.isdir(_rootdir) and _rootdir.endswith('pygodot.resources')):
    sys.path.append(_rootdir)
    inside_godot = True

if inside_godot:
    from utils import _pyprint as print
else:
    print = print

__version__ = get_version()
