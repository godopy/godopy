from godot_cpp cimport UtilityFunctions

import sys

class Writer:
    def write(self, data):
        if isinstance(data, str):
            msg = data.rstrip().encode('utf-8')
        else:
            msg = bytes(data).rstrip()
        UtilityFunctions.print(<const char *>msg)
    
    def flush(self): pass

class ErrWriter:
    def write(self, data):
        if isinstance(data, str):
            msg = data.rstrip().encode('utf-8')
        else:
            msg = bytes(data).rstrip()
        UtilityFunctions.print(<const char *>msg)

    def flush(self): pass

sys.stdout = Writer()
sys.stderr = ErrWriter()

