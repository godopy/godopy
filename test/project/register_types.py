import sys
import math
import godot as gd
import _gdextension as gde
# from _vector_types import Vector2

GDExample = gde.ExtensionClass('GDExample', 'Sprite2D')

# @GDExample.add_method
def __init__(self):
    self.time_passed: float = 0.0

# @GDExample.add_method
def _process(self, delta: float) -> None:
    self.time_passed += delta
    new_position = (
        10.0 + (10.0 * math.sin(self.time_passed * 2.0)),
        10.0 + (10.0 * math.cos(self.time_passed * 1.5))
    )

    set_position = gd.MethodBind(self, 'set_position')
    set_position(new_position)

def initialize():
    lines = ["Active Python paths:"]
    for path in sys.path:
        lines.append('\t[color=green]%s[/color]' % path)
    # TODO: Only in verbose mode (OS.is_stdout_verbose)
    gd.print_rich('[color=gray]%s[/color]\n' % '\n'.join(lines))


GDExample.add_methods(__init__, _process)


def register():
    GDExample.register()
