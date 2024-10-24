from binding cimport *
from godot_cpp cimport Variant, String, UtilityFunctions, OS, Engine, ProjectSettings
from gdextension cimport (
    ExtensionClass,
    _registered_classes,
    _NODEDB,
    _OBJECTDB,
    _METHODDB,
    _BUILTIN_METHODDB,
    _CLASSDB
)


import io
import os
import gc
import sys
import traceback
import importlib

import default_gdextension_config

try:
    mod = os.environ['GDEXTENSION_CONFIG_MODULE']
    gdextension_config = importlib.import_module(mod)
except (KeyError, ImportError):
    try:
        import gdextension_config
    except ImportError:
        gdextension_config = default_gdextension_config

# gdextension_config = default_gdextension_config

cdef tuple registered_packages = tuple(gdextension_config.REGISTERED_PACKAGES)


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


cdef object initialize_funcs = []
cdef object deinitialize_funcs = []
cdef bint is_first_level = True
cdef int first_level = 0


cdef public int python_initialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    with gil:
        _python_initialize_level(p_level)

    return 0


cdef public int python_deinitialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    with gil:
        _python_deinitialize_level(p_level)

    return 0



cdef void _python_initialize_level(ModuleInitializationLevel p_level) except *:
    global initialize_funcs, deinitialize_funcs, is_first_level, first_level, registered_packages

    for module_name in registered_packages:
        mod_register_types = importlib.import_module('%s.register_types' % module_name)
        initialize_func = getattr(mod_register_types, 'initialize', None)
        deinitialize_func = getattr(mod_register_types, 'uninitialize', None)
        if deinitialize_func is None:
            deinitialize_func = getattr(mod_register_types, 'deinitialize', None)
        if initialize_func is not None:
            initialize_funcs.append(initialize_func)
        if deinitialize_func is not None:
            deinitialize_funcs.append(deinitialize_func)

    UtilityFunctions.print_verbose("GodoPy Python initialization started, level %d" % p_level)

    if is_first_level:
        redirect_python_stdio()

        if ProjectSettings.get_singleton().has_setting("application/run/print_header"):
            UtilityFunctions.print("GodoPy version %s" % '0.0.0.5dev')
            UtilityFunctions.print("Python version %s\n" % sys.version)

        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path and Engine.get_singleton().is_editor_hint():
            sys.path.append(os.path.join(venv_path, 'Lib', 'site-packages'))

        try:
            import register_types
            initialize_func = getattr(register_types, 'initialize', None)
            deinitialize_func = getattr(register_types, 'uninitialize', None)
            if deinitialize_func is None:
                deinitialize_func = getattr(register_types, 'deinitialize', None)
            if initialize_func is not None:
                initialize_funcs.append(initialize_func)
            if deinitialize_func is not None:
                deinitialize_funcs.append(deinitialize_func)
        except ImportError as exc:
            f = io.StringIO()
            traceback.print_exception(exc, file=f)
            exc_text = f.getvalue()
            if isinstance(exc, ModuleNotFoundError) and "'register_types'" in exc_text:
                UtilityFunctions.print_rich(
                    "\n[color=orange]WARNING: 'register types' module was not found.[/color]\n"
                )
            else:
                raise exc

        is_first_level = False
        first_level = p_level
        deinitialize_funcs.reverse()

        # gc.set_debug(gc.DEBUG_LEAK)

    for func in initialize_funcs:
        func(p_level)


cdef void _python_deinitialize_level(ModuleInitializationLevel p_level) except *:
    global deinitialize_funcs, _registered_classes
    cdef ExtensionClass cls

    UtilityFunctions.print_verbose("GodoPy Python cleanup, level %d" % p_level)

    for func in deinitialize_funcs:
        func(p_level)

    if p_level == first_level:
        _NODEDB = {}
        _OBJECTDB = {}
        _METHODDB = {}
        _BUILTIN_METHODDB = {}
        _CLASSDB = {}

        for cls in _registered_classes:
            cls.unregister()

        _registered_classes = []

        gc.collect()
