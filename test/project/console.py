import sys
import atexit
import readline
import rlcompleter

from code import InteractiveConsole

import godot as gd

class GodotInteractiveConsole(InteractiveConsole):
    def __init__(self):
        super().__init__({
            "__name__": "__console__",
            "__doc__": None
        })
        # OS = gd.GodotSingleton('OS')
        # self._godot_input = gd.GodotMethodBindRet(OS, 'read_string_from_stdin', 2841200299, 'String')

    def interact(self, banner, exitmsg=None):
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = ">>> "
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = "... "

        readline.parse_and_bind("tab: complete")
        readline.set_completer(rlcompleter.Completer().complete)

        # Release references early at shutdown (the readline module's
        # contents are quasi-immortal, and the completer function holds a
        # reference to globals).
        atexit.register(lambda: readline.set_completer(None))

        self.write("%s\n\n" % str(banner))
        more = 0
        while 1:
            try:
                if more:
                    prompt = sys.ps2
                else:
                    prompt = sys.ps1
                try:
                    line = self.raw_input( prompt)
                except EOFError:
                    self.write("\n")
                    break
                else:
                    more = self.push(line)
            except KeyboardInterrupt:
                self.write("\nKeyboardInterrupt\n")
                self.resetbuffer()
                more = 0
        if exitmsg is None:
            self.write('now exiting %s...' % self.__class__.__name__)
        elif exitmsg != '':
            self.write('%s\n' % exitmsg)

    def write(self, data):
        gd.printraw(data)

    def raw_input(self, prompt=None):
        return gd.input(prompt)


def interact(godot_banner=None):
    console = GodotInteractiveConsole()
    if godot_banner is None:
        godot_banner = console.__class__.name

    console.interact(
        banner='Python %s on %s\n[%s]' % \
               (sys.version, sys.platform, godot_banner)
    )
