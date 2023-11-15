'''
Custom stdout/stderr writers which
redirect output to Godot utility functions
'''
from godot_cpp cimport UtilityFunctions

import sys

class Writer:
    def write(self, data):
        if isinstance(data, str):
            msg = data.encode('utf-8')
        else:
            msg = bytes(data)
        UtilityFunctions.printraw(<const char *>msg)
    
    def flush(self): pass

class ErrWriter:
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

sys.stdout = Writer()
sys.stderr = ErrWriter()

