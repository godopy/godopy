import sys
import numpy as np
import godot as gd
from godot import extension as gde
# from _vector_types import Vector2


def initialize(level):
    if level != 2:
        return

    lines = ["Active Python paths:"]
    for path in sys.path:
        lines.append('\t[color=green]%s[/color]' % path)
    # TODO: Only in verbose mode (OS.is_stdout_verbose)
    gd.print_rich('[color=gray]%s[/color]\n' % '\n'.join(lines))

    gdexample().register()
    console().register()


def gdexample():
    GDExample = gde.ExtensionClass('GDExample', 'Sprite2D')

    def __init__(self):
        self.time_passed: float = 0.0

    def _process(self, delta: float) -> None:
        self.time_passed += delta
        new_position = (
            10.0 + (10.0 * np.sin(self.time_passed * 2.0)),
            10.0 + (10.0 * np.cos(self.time_passed * 1.5))
        )

        set_position = gd.MethodBind(self, 'set_position')
        set_position(new_position)

    GDExample.add_methods(__init__, _process)

    return GDExample


def console():
    TerminalConsole = gde.ExtensionClass('TerminalConsole', 'SceneTree')

    def _init(self):
        from godopy.contrib.console import terminal

        terminal.interact()

        quit = gd.MethodBind(self, 'quit')
        quit()

    TerminalConsole.add_methods(_init)

    return TerminalConsole
