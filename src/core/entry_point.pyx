"""
Extends GDExtension Initialization API. Enables initialization of Python
modules.
"""

from binding cimport *
from godot_cpp cimport Variant, String, UtilityFunctions, OS, Engine, ProjectSettings
from gdextension cimport ExtensionClass,  _registered_classes, configure

from godot_types cimport string_from_pyobject
from _gdextension_internals cimport print_traceback

import io
import os
import sys
import traceback
import importlib

import _gdextension_internals


__all__ = ['get_config']


try:
    mod = os.environ['GDEXTENSION_CONFIG_MODULE']
    gdextension_config_module = importlib.import_module(mod)
except (KeyError, ImportError):
    try:
        import gdextension_config as gdextension_config_module
    except ImportError:
        gdextension_config_module = None


cdef class Config:
    cdef dict _data

    def __cinit__(self):
        self._data = {}

    def get(self, attr):
        return self._data[attr]

    def set(self, attr, value):
        self._data[attr] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def update(self, other):
        self._data.update(other)


cdef Config _config = Config()


def get_config() -> Config:
    return _config


if gdextension_config_module is None:
    _config.update(_gdextension_internals.default_gdextension_config)
else:
    for key in ('REGISTERED_MODULES',):
        _config[key.lower()] = gdextension_config_module.__dict__.get(
            key, _gdextension_internals.default_gdextension_config.get(key.lower())
        )

# Defaults from gdextension module
configure(_config)


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
cdef object uninitialize_funcs = []
cdef object configure_funcs = []

cdef bint is_first_level = True
cdef int first_level = 0


cdef public int python_initialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    with gil:
        try:
            _python_initialize_level(p_level)
        except Exception as exc:
            print_traceback(exc)

    return 0


cdef public int python_deinitialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    with gil:
        try:
            _python_deinitialize_level(p_level)
        except Exception as exc:
            print_traceback(exc)
    
    return 0


cdef void _python_initialize_level(ModuleInitializationLevel p_level) except *:
    global initialize_funcs, uninitialize_funcs, configure_funcs, is_first_level, first_level, \
           gdextension_config, config

    UtilityFunctions.print_verbose("GodoPy Python initialization started, level %d" % p_level)

    for module_name in _config.get('registered_modules'):
        mod_register_types = importlib.import_module('%s.register_types' % module_name)

        initialize_func = getattr(mod_register_types, 'initialize', None)
        uninitialize_func = getattr(mod_register_types, 'uninitialize', None)
        configure_func = getattr(mod_register_types, 'configure', None)

        if initialize_func is not None:
            initialize_funcs.append(initialize_func)

        if uninitialize_func is not None:
            uninitialize_funcs.append(uninitialize_func)

        if configure_func is not None:
            configure_funcs.append(configure_func)


    if is_first_level:
        redirect_python_stdio()

        if ProjectSettings.get_singleton().has_setting("application/run/print_header"):
            UtilityFunctions.print("GodoPy version %s" % '0.1dev')
            UtilityFunctions.print("Python version %s\n" % sys.version)

        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path and Engine.get_singleton().is_editor_hint():
            sys.path.append(os.path.join(venv_path, 'Lib', 'site-packages'))

        try:
            import register_types

            initialize_func = getattr(register_types, 'initialize', None)
            uninitialize_func = getattr(register_types, 'uninitialize', None)
            configure_fuc = getattr(register_types, 'configure', None)

            if initialize_func is not None:
                initialize_funcs.append(initialize_func)

            if uninitialize_func is not None:
                uninitialize_funcs.append(uninitialize_func)

            if configure_func is not None:
                configure_funcs.append(configure_func)


        except ImportError as exc:
            f = io.StringIO()
            traceback.print_exception(exc, file=f)
            exc_text = f.getvalue()
            if isinstance(exc, ModuleNotFoundError) and "'register_types'" in exc_text:
                pass
            else:
                raise exc

        for configure in configure_funcs:
            configure(_config)

        is_first_level = False
        first_level = p_level
        uninitialize_funcs.reverse()

    for func in initialize_funcs:
        func(p_level)


cdef void _python_deinitialize_level(ModuleInitializationLevel p_level) except *:
    global uninitialize_funcs, _registered_classes, first_level
    cdef ExtensionClass cls

    UtilityFunctions.print_verbose("GodoPy Python cleanup, level %d" % p_level)

    for func in uninitialize_funcs:
        func(p_level)

    if p_level == first_level:
        for cls in _registered_classes:
            cls.unregister()

        _registered_classes = []
