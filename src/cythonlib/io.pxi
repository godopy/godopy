cdef UtilityFunctionNoRet _printraw = UtilityFunctionNoRet('printraw', 2648703342)
cdef UtilityFunctionNoRet _print_rich = UtilityFunctionNoRet('print_rich', 2648703342)
cdef UtilityFunctionNoRet _push_error = UtilityFunctionNoRet('push_error', 2648703342)
cdef UtilityFunctionNoRet _push_warning = UtilityFunctionNoRet('push_warning', 2648703342)

print = UtilityFunctionNoRet('print', 2648703342)
printerr = UtilityFunctionNoRet('printerr', 2648703342)
printt = UtilityFunctionNoRet('printt', 2648703342)
prints = UtilityFunctionNoRet('prints', 2648703342)
printverbose = UtilityFunctionNoRet('print_verbose', 2648703342)

print_rich = _print_rich
printraw = _printraw
push_error = _push_error
push_warningg = _push_warning

_OS = Singleton('OS')
_OS__read_string_from_stdin = MethodBindRet(_OS, 'read_string_from_stdin', 2841200299, 'String')

cpdef input(str prompt=None):
    if prompt is not None:
        printraw(prompt)

    return _OS__read_string_from_stdin()


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
