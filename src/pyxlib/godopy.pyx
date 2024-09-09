
from godot_cpp cimport UtilityFunctions

import sys

'''
Custom stdout/stderr writers which
redirect output to Godot utility functions
'''
class GodotWriter:
    def write(self, data):
        if isinstance(data, str):
            msg = data.encode('utf-8')
        else:
            msg = bytes(data)
        UtilityFunctions.printraw(<const char *>msg)

    def flush(self): pass

class GodotErrWriter:
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

def init_godot_printing():
    sys.stdout = GodotWriter()
    sys.stderr = GodotErrWriter()

