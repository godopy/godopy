'''
Custom stdout/stderr writers which
redirect output to Godot utility functions
'''
cdef class StdoutWriter:
    cpdef write(self, data):
        godot.printraw(str(data))

    cpdef flush(self): pass

    cpdef fileno(self):
        return 1

cdef class StderrWriter:
    cdef list data

    def __cinit__(self):
        self.data = []

    cpdef write(self, object data):
        self.data.append(str(data))

    cpdef flush(self):
        msg = '[color=red]%s[/color]' % (''.join(self.data))
        godot.print_rich(msg)
        self.data = []

    cpdef fileno(self):
        return 2


def redirect_python_stdout():
    sys.stdout = StdoutWriter()
    sys.stderr = StderrWriter()