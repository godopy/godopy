from godot_cpp cimport UtilityFunctions

import io, traceback


cdef public int print_traceback(object exc) except -1:
    cdef object f = io.StringIO()

    traceback.print_exception(exc, file=f)
    cdef object exc_text = f.getvalue()

    UtilityFunctions.print_rich(
        "\n[color=red]ERROR: %s[/color]"
        "\n[color=orange]%s[/color]\n" % (exc, exc_text)
    )

    return 0


default_gdextension_config = {
    'registered_modules': ('godot', 'godot_scripting', 'godopy'),
}
