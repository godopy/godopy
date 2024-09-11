'''
Custom stdout/stderr writers which
redirect output to Godot utility functions
'''
cdef class StdoutWriter:
    cpdef write(self, data):
        if isinstance(data, str):
            msg = data.encode('utf-8')
        else:
            msg = bytes(data)
        printraw(msg)

    cpdef flush(self): pass

cdef class StderrWriter:
    cdef list data

    def __cinit__(self):
        self.data = []

    cpdef write(self, object data):
        if isinstance(data, str):
            msg = data.encode('utf-8')
        else:
            msg = bytes(data)
        self.data.append(msg)

    cpdef flush(self):
        msg = b'[color=red]%s[/color]' % (b''.join(self.data))
        print_rich(msg)
        self.data = []

def redirect_python_stdout():
    sys.stdout = StdoutWriter()
    sys.stderr = StderrWriter()
