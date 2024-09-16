import sys
import _godot as gd

def initialize():
    lines = ["Active Python paths:"]
    for path in sys.path:
        lines.append('\t[color=green]%s[/color]' % path)
    gd.print_rich('[color=gray]%s[/color]\n' % '\n'.join(lines))
