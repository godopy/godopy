from .headers.gdnative_api cimport godot_string, godot_object
from .globals cimport gdapi
from ._core cimport _Wrapped

from libc.stddef cimport wchar_t
from cpython.mem cimport PyMem_Free

cdef extern from "Python.h":
    cdef wchar_t *Py_GetPythonHome()
    cdef char *Py_EncodeLocale(const wchar_t *text, size_t *error_pos)

import os
import sys

def _pyprint(*objects, sep=' ', end='\n'):
    _gdprint(sep.join(str(o) for o in objects))


def _gdprint(str fmt, *args):
    cdef msg = fmt.format(*args).encode('utf-8')
    cdef const char *c_msg = msg
    cdef godot_string gd_msg

    gdapi.godot_string_new(&gd_msg)
    gdapi.godot_string_parse_utf8(&gd_msg, c_msg)
    gdapi.godot_print(&gd_msg)
    gdapi.godot_string_destroy(&gd_msg)


cdef public _Wrapped _create_wrapper(godot_object *_owner, size_t _type_tag):
    cdef _Wrapped wrapper = _Wrapped.__new__(_Wrapped)
    wrapper._owner = _owner
    wrapper._type_tag = _type_tag
    print('Godot wrapper %s created' % wrapper)
    return wrapper


_dynamic_loading_initialized = False

def _init_dynamic_loading():
    global _dynamic_loading_initialized

    if _dynamic_loading_initialized:
        return

    cdef bytes b_home
    cdef char *c_home = Py_EncodeLocale(Py_GetPythonHome(), NULL)
    try:
        b_home = c_home
    finally:
        PyMem_Free(<void *>c_home)

    home = b_home.decode('utf-8')

    dirname, base = os.path.split(home)
    godot_path = detect_godot_project(dirname, base)
    _pyprint('GODOT PROJECT PATH:', godot_path)

    if not godot_path:
        raise RuntimeError('Could not detect the Godot project. Please check the location of the PyGodot library.')

    project_path, godot_project_name = os.path.split(godot_path)

    _pyprint('PYGODOT PROJECT PATH:', project_path)

    def fullpath(*args):
        return os.path.join(project_path, *args)

    def file_exists(*args):
        return os.path.exists(fullpath(*args))

    if not file_exists('gdlibrary.py'):
        raise RuntimeError('No "gdlibrary.py" found. PyGodot initialization aborted.')

    sys.path.insert(0, project_path)

    if not file_exists('__init__.py'):
        with open(fullpath('__init__.py'), 'w'):
            pass

    import gdlibrary

    if not hasattr(gdlibrary, 'nativescript_init'):
        raise RuntimeError('GDLibrary doesn\'t provide NativeScript initialization routine. PyGodot initialization aborted.')

    _dynamic_loading_initialized = True
    gdlibrary.nativescript_init()

cdef str detect_godot_project(str dirname, str base):
    if not dirname or not base:
        return ''

    if 'project.godot' in os.listdir(dirname):
        return dirname

    dirname, base = os.path.split(dirname)
    return detect_godot_project(dirname, base)
