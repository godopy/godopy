import sys
from code import InteractiveConsole
import godot as gd

class GodotTerminalConsole(InteractiveConsole):
    def __init__(self):
        super().__init__({
            "__name__": "__console__",
            "__doc__": None,
            "print_rich": gd.print_rich
        })

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
            import atexit
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
                self.resetbuffer()
                more = 0

    def write(self, data):
        gd.printraw(data)

    def raw_input(self, prompt=None):
        return gd.input(prompt)


def interact(godot_version=None):
    console = GodotTerminalConsole()

    banner = [f'\n| Python version {sys.version}\n']
    if godot_version is not None:
        banner += [
            '| Godot Engine version %(major)s.%(minor)s.%(status)s.%(build)s.' % godot_version,
            f'{godot_version['hash'][:9]}\n',
        ]
    banner += ['| Interactive Console']

    console.interact(banner=''.join(banner))
