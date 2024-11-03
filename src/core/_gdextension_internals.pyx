from libc.stdint cimport int32_t, uint64_t
from cpython cimport PyObject
from binding cimport (
    gdextension_interface_print_error as print_error
)
from godot_cpp cimport UtilityFunctions, OS, String

import io, traceback


cdef int print_traceback(object exc) except -1:
    cdef object f = io.StringIO()

    traceback.print_exception(exc, file=f)
    cdef object exc_text = f.getvalue()

    exc_lines = exc_text.splitlines()
    info_line_str = exc_lines[-2]

    if info_line_str.strip().startswith('^'):
        info_line_str = exc_lines[-4]

    info_line = [s.strip().strip(',') for s in info_line_str.split()]

    cdef bytes descr = str(exc).encode('utf-8')
    cdef bytes path = info_line[1].encode('utf-8')
    cdef bytes func = info_line[-1].encode('utf-8')
    cdef int32_t line = int(info_line[3])

    UtilityFunctions.print_rich("[color=purple]%s[/color]" % exc_text)
    print_error(descr, path, func, line, False)

    return 0


cdef public void print_traceback_and_die(object exc) noexcept:
    print_traceback(exc)

    # Kill the process immediately to prevent a flood of unrelated errors
    cdef uint64_t pid = OS.get_singleton().get_process_id()
    OS.get_singleton().kill(pid)


default_gdextension_config = {
    'registered_modules': ('godot', 'godot_scripting', 'godopy'),
}
