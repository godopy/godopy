import sys

import godot
from gdexample import GDExample


def initialize(level):
    if level == godot.MODULE_INITIALIZATION_LEVEL_SCENE:
        GDExample.register()

        lines = ["[color=gray]Active Python paths:[/color]"]
        for path in sys.path:
            lines.append('\t[color=green]%s[/color]' % path)

        godot.print_rich('%s\n' % '\n'.join(lines))
