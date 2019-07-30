from godot_headers.gdnative_api cimport *
from godot.globals cimport gdapi
from godot.bindings._core cimport _Wrapped

from libc.stddef cimport wchar_t
from cpython.mem cimport PyMem_Free

import os
import sys


cdef extern from "Python.h":
    cdef wchar_t *Py_GetPythonHome()
    cdef char *Py_EncodeLocale(const wchar_t *text, size_t *error_pos)


cdef bytes godot_string_to_bytes(const godot_string *s):
    cdef godot_char_string chars = gdapi.godot_string_utf8(s)
    cdef const char *cs = gdapi.godot_char_string_get_data(&chars)
    cdef bytes bs
    try:
        bs = cs
    finally:
        gdapi.godot_char_string_destroy(&chars)
    return bs


cdef bint _dynamic_loading_initialized = False


cdef str godot_project_dir():
    cdef bytes b_home
    cdef const char *c_home = Py_EncodeLocale(Py_GetPythonHome(), NULL)
    try:
        b_home = c_home
    finally:
        PyMem_Free(<void *>c_home)

    home = b_home.decode('utf-8')

    dirname, base = os.path.split(home)
    return _detect_godot_project(dirname, base)


cdef int _init_dynamic_loading() except -1:
    global _dynamic_loading_initialized

    if _dynamic_loading_initialized:
        return 0

    godot_path = godot_project_dir()
    print('GODOT PROJECT PATH:', godot_path)

    if not godot_path:
        raise RuntimeError('Could not detect the Godot project. Please check the location of the PyGodot library.')

    project_path, godot_project_name = os.path.split(godot_path)

    def fullpath(*args):
        return os.path.join(project_path, *args)

    def file_exists(*args):
        return os.path.exists(fullpath(*args))

    if not file_exists('gdlibrary.py'):
        raise RuntimeError('No "gdlibrary.py" found. PyGodot initialization aborted.')

    sys.path.insert(0, project_path)

    _dynamic_loading_initialized = True
    return 0


cdef str _detect_godot_project(str dirname, str base):
    if not dirname or not base:
        return ''

    if 'project.godot' in os.listdir(dirname):
        return dirname

    dirname, base = os.path.split(dirname)
    return _detect_godot_project(dirname, base)
