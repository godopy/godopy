from .headers.gdnative_api cimport godot_string, godot_char_string, godot_object
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
    cdef bytes msg = sep.join(str(o) for o in objects).encode('utf-8')
    cdef const char *c_msg = msg
    cdef godot_string gd_msg

    gdapi.godot_string_new(&gd_msg)
    gdapi.godot_string_parse_utf8(&gd_msg, c_msg)
    gdapi.godot_print(&gd_msg)
    gdapi.godot_string_destroy(&gd_msg)

def _gdprint(str fmt, *args):
    cdef bytes msg = fmt.format(*args).encode('utf-8')
    cdef const char *c_msg = msg
    cdef godot_string gd_msg

    gdapi.godot_string_new(&gd_msg)
    gdapi.godot_string_parse_utf8(&gd_msg, c_msg)
    gdapi.godot_print(&gd_msg)
    gdapi.godot_string_destroy(&gd_msg)


cdef bytes godot_string_to_bytes(const godot_string *s):
    cdef godot_char_string chars = gdapi.godot_string_utf8(s)
    cdef const char *cs = gdapi.godot_char_string_get_data(&chars)
    cdef bytes bs
    try:
        bs = cs
    finally:
        gdapi.godot_char_string_destroy(&chars)
    return bs

cdef public _Wrapped _create_wrapper(godot_object *_owner, size_t _type_tag):
    cdef _Wrapped wrapper = _Wrapped.__new__(_Wrapped)
    wrapper._owner = _owner
    wrapper._type_tag = _type_tag
    print('Godot wrapper %s created' % wrapper)
    return wrapper


cdef public int _nativescript_python_init() except -1:
    if _init_dynamic_loading() != 0: return -1

    import gdlibrary

    if hasattr(gdlibrary, 'nativescript_init'):
        gdlibrary.nativescript_init()

    return 0

cdef public int _gdnative_python_singleton() except -1:
    if _init_dynamic_loading() != 0: return -1

    import gdlibrary

    if hasattr(gdlibrary, 'gdnative_singleton'):
        gdlibrary.gdnative_singleton()

    return 0

cdef bint _dynamic_loading_initialized = False

cdef int _init_dynamic_loading() except -1:
    global _dynamic_loading_initialized

    if _dynamic_loading_initialized:
        return 0

    cdef bytes b_home
    cdef const char *c_home = Py_EncodeLocale(Py_GetPythonHome(), NULL)
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

    _dynamic_loading_initialized = True
    return 0
    

cdef str detect_godot_project(str dirname, str base):
    if not dirname or not base:
        return ''

    if 'project.godot' in os.listdir(dirname):
        return dirname

    dirname, base = os.path.split(dirname)
    return detect_godot_project(dirname, base)
