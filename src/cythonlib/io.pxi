cdef UtilityFunction _printraw = UtilityFunction('printraw')
cdef UtilityFunction _print_rich = UtilityFunction('print_rich')
cdef UtilityFunction _push_error = UtilityFunction('push_error')
cdef UtilityFunction _push_warning = UtilityFunction('push_warning')

print = UtilityFunction('print')
printerr = UtilityFunction('printerr')
printt = UtilityFunction('printt')
prints = UtilityFunction('prints')
printverbose = UtilityFunction('print_verbose')

print_rich = _print_rich
printraw = _printraw
push_error = _push_error
push_warningg = _push_warning


cpdef input(str prompt=None):
    if prompt is not None:
        printraw(prompt)

    return OS.get_singleton().read_string_from_stdin()


cdef class StdoutWriter:
    cpdef write(self, data):
        _printraw(str(data))

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
        _print_rich(msg)
        self.data = []

    cpdef fileno(self):
        return 2


def redirect_python_stdio():
    sys.stdout = StdoutWriter()
    sys.stderr = StderrWriter()
    # TODO: stdin via custom class with _input
