from binding cimport *
from godot_cpp cimport UtilityFunctions, OS, Engine

import io
import os
import sys
import traceback


cdef class StdoutWriter:
    cpdef write(self, data):
        UtilityFunctions.printraw(str(data))

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
        # msg = '[color=red]%s[/color]' % (''.join(self.data))
        UtilityFunctions.printerr(''.join(self.data))
        self.data = []

    cpdef fileno(self):
        return 2


def redirect_python_stdio():
    sys.stdout = StdoutWriter()
    sys.stderr = StderrWriter()


cdef object initialize_func = None
cdef object deinitialize_func = None
cdef bint first_level = True


try:
    from godot import register_types as godot_register_types
except ImportError as exc:
    UtilityFunctions.print_rich(
        "\n[color=red]ERROR: 'godot.register types' module was not found or raised an exception.[/color]\n"
    )
    raise SystemError("GodoPy was not properly installed, 'godot.register_types' is missing")

try:
    from godopy import register_types as godopy_register_types
except ImportError as exc:
    UtilityFunctions.print_rich(
        "\n[color=red]ERROR: 'godopy.register types' module was not found or raised an exception.[/color]\n"
    )
    raise SystemError("GodoPy was not properly installed, 'godopy.register_types' is missing")


cdef public int python_initialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    return _python_initialize_level(p_level)


cdef public int python_deinitialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    return _python_deinitialize_level(p_level)


cdef int _python_initialize_level(ModuleInitializationLevel p_level) noexcept with gil:
    global initialize_func, deinitialize_func, first_level

    UtilityFunctions.print_verbose("GodoPy Python initialization started, level %d" % p_level)

    if first_level:
        redirect_python_stdio()

        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path and Engine.get_singleton().is_editor_hint():
            sys.path.append(os.path.join(venv_path, 'Lib', 'site-packages'))

        try:
            import register_types
            initialize_func = getattr(register_types, 'initialize', None)
            deinitialize_func = getattr(register_types, 'deinitialize', None)
        except ImportError as exc:
            f = io.StringIO()
            traceback.print_exception(exc, file=f)
            exc_text = f.getvalue()
            if isinstance(exc, ModuleNotFoundError) and "'register_types'" in exc_text:
                UtilityFunctions.print_rich(
                    "\n[color=orange]WARNING: 'register types' module was not found.[/color]\n"
                )
            else:
                UtilityFunctions.print_rich(
                    "\n[color=red]ERROR: 'register types' module raised an exception:[/color]"
                    "\n[color=orange]%s[/color]\n" % exc_text
            )

        first_level = False

    godot_register_types.initialize(p_level)
    godopy_register_types.initialize(p_level)

    if initialize_func is not None:
        initialize_func(p_level)

    return 0


cdef int _python_deinitialize_level(ModuleInitializationLevel p_level) noexcept with gil:
    global deinitialize_func

    UtilityFunctions.print_verbose("GodoPy Python cleanup, level %d" % p_level)

    if deinitialize_func:
        deinitialize_func(p_level)

    godopy_register_types.deinitialize(p_level)
    godot_register_types.deinitialize(p_level)

    return 0
