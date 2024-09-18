import sys
import atexit  
from code import InteractiveConsole

import _godot as gd

class GodotInteractiveConsole(InteractiveConsole):
    def __init__(self):
        super().__init__({
            "__name__": "__console__",
            "__doc__": None,
            "print_rich": gd.print_rich
        })
        # OS = gd.GodotSingleton('OS')
        # self._godot_input = gd.GodotMethodBindRet(OS, 'read_string_from_stdin', 2841200299, 'String')

    def interact(self, banner):
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = ">>> "
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = "... "

        try:
            import readline
            import rlcompleter
            readline.parse_and_bind("tab: complete")
            readline.set_completer(rlcompleter.Completer().complete)

            # Release references early at shutdown (the readline module's
            # contents are quasi-immortal, and the completer function holds a
            # reference to globals).
            atexit.register(lambda: readline.set_completer(None))
        except ImportError:
            pass

        self.write("%s\n\n" % str(banner))
        more = 0
        while 1:
            try:
                if more:
                    prompt = sys.ps2
                else:
                    prompt = sys.ps1

                line = self.raw_input(prompt)
                if line.strip() in ['exit', 'quit']:
                    return
                else:
                    more = self.push(line)
            except KeyboardInterrupt:
                return

    def write(self, data):
        gd.printraw(data)

    def raw_input(self, prompt=None):
        return gd.input(prompt)


def interact(godot_version=None):
    console = GodotInteractiveConsole()

    console.interact(
        banner=''.join([
            f'| Python version {sys.version}\n',
             '| Godot Engine version %(major)s.%(minor)s.%(status)s.%(build)s.' % godot_version,
            f'{godot_version['hash'][:9]}\n',
            '| Interactive Console'
        ])
    )
