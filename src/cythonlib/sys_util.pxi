'''
Custom stdout/stderr writers which
redirect output to Godot utility functions
'''
class StdoutWriter:
    def write(self, data):
        if isinstance(data, str):
            msg = data.encode('utf-8')
        else:
            msg = bytes(data)
        UtilityFunctions.printraw(<const char *>msg)

    def flush(self): pass

class StderrWriter:
    def __init__(self):
        self.data = []

    def write(self, data):
        if isinstance(data, str):
            msg = data.encode('utf-8')
        else:
            msg = bytes(data)
        self.data.append(msg)

    def flush(self):
        # TODO: Use push_error in custom handler
        msg = b'[color=red]%s[/color]' % (b''.join(self.data))
        UtilityFunctions.print_rich(msg)
        self.data = []

def redirect_python_stdout():
    sys.stdout = StdoutWriter()
    sys.stderr = StderrWriter()
