import sys
import godot as gd


def initialize(level):
    if level != 2:
        return

    lines = ["[color=gray]Active Python paths:[/color]"]
    for path in sys.path:
        lines.append('\t[color=green]%s[/color]' % path)

    gd.print_rich('%s\n' % '\n'.join(lines))

