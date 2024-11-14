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

    cdef bytes descr, path, func
    cdef int32_t line

    try:
        descr = str(exc).encode('utf-8')

        exc_lines = exc_text.splitlines()
        if len(exc_lines) > 1:
            info_line_str = exc_lines[-2]

            if info_line_str.lstrip().startswith('^'):
                info_line_str = exc_lines[-4]
            elif not (info_line_str.lstrip().startswith('File') and 'line' in info_line_str):
                info_line_str = exc_lines[-3]

            info_line = [s.strip().strip(',') for s in info_line_str.split()]

            path = info_line[1].encode('utf-8')
            func = info_line[-1].encode('utf-8')
            line = int(info_line[3])
        else:
            info_line_str = exc_lines[0]
            info_line = [s.strip().strip(':') for s in info_line_str.split()]
            path_line = info_line[1].rsplit(':', 1)
            path = path_line[0].encode('utf-8')
            func = b''
            line = int(path_line[1])

        UtilityFunctions.print_rich("[color=purple]%s[/color]" % exc_text)
        print_error(descr, path, func, line, False)

    except Exception as exc2:
        UtilityFunctions.print_rich("[color=red]Exception parser raised an exception: %s[/color]" % exc2)

        UtilityFunctions.print_rich("[color=purple]%s[/color]" % exc_text)
        UtilityFunctions.push_error(str(exc))

    return 0


cdef public void print_traceback_and_die(object exc) noexcept:
    print_traceback(exc)

    # Kill the process immediately to prevent a flood of unrelated errors
    cdef uint64_t pid = OS.get_singleton().get_process_id()
    OS.get_singleton().kill(pid)


default_gdextension_config = {
    'registered_modules': ('godot', 'godot_scripting', 'godopy'),
}
