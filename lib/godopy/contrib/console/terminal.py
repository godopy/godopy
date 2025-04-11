import os
import sys
from code import InteractiveConsole

import godot
from godot.classdb import Engine, OS

BUFFER_SIZE = 2048


rl = None


def exit():
    raise EOFError()


class GodotTerminalConsole(InteractiveConsole):
    def __init__(self):
        super().__init__(
            {"__name__": "__console__", "__doc__": None, "exit": exit, "quit": exit}
        )

    def runcode(self, code):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to
        display a traceback.  All exceptions are caught except
        SystemExit, which is reraised.

        A note about KeyboardInterrupt: this exception may occur
        elsewhere in this code, and may not always be caught.  The
        caller should be prepared to deal with it.

        """
        try:
            exec(code, self.locals)
        except (SystemExit, EOFError):
            raise
        except:
            self.showtraceback()

    def interact(self, banner):
        global rl
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = ">>> "
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = "... "

        venv_path = os.environ.get("VIRTUAL_ENV")
        if venv_path:
            sys.path.append(
                os.path.join(venv_path, "lib", "python3.12", "site-packages")
            )
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

            rl = readline.rl
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

                if rl is not None:
                    line = rl.readline(prompt)
                    if line.strip() in ["exit", "quit"]:
                        raise EOFError()

                    more = self.push(line)

                else:
                    line = self.raw_input(prompt)

                    if line.strip() in [b"exit", b"quit"]:
                        raise EOFError()
                    else:
                        more = self.push(line.decode("utf-8", errors="replace"))

            except EOFError:
                return

            except KeyboardInterrupt:
                self.resetbuffer()
                more = 0

    def write(self, data):
        godot.printraw(data)

    def raw_input(self, prompt=None):
        if prompt:
            godot.printraw(prompt)

        buffer = OS.read_buffer_from_stdin(BUFFER_SIZE)
        first_byte = buffer[0]

        if first_byte in (4, 45):
            raise EOFError()

        return buffer.tobytes().split(b"\n", 1)[0].rstrip(b"\r")


def interact():
    console = GodotTerminalConsole()

    banner = [f"\n| Python version {sys.version}\n"]
    godot_version = Engine.get_version_info()

    banner += [
        "| Godot Engine version %(major)s.%(minor)s.%(status)s.%(build)s."
        % godot_version,
        f"{godot_version['hash'][:9]}\n",
    ]
    banner += ["| Interactive Console"]

    console.interact(banner="".join(banner))
