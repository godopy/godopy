"""\
This module wraps GodoPy extensions to the engine
and holds the extension runtime data
"""
cimport cython
from cpython cimport ref, PyObject
from libcpp.vector cimport vector
from godot_cpp cimport *
from cython.operator cimport dereference as deref

cimport godot

import sys
import types

include "gde_shortcuts.pxi"
include "type_converters.pxi"

include "extension.pxi"
include "extension_class.pxi"
include "extension_method.pxi"

include "console.pxi"

cdef object init_func = None
cdef object register_func = None
cdef object unregister_func = None
cdef object terminate_func = None


def initialize_types():
    global init_func, register_func, unregister_func, terminate_func

    redirect_python_stdio()

    # print("Python-level initialization started")
    register_func = None
    try:
        import register_types
        init_func = getattr(register_types, 'initialize', None)  
        register_func = getattr(register_types, 'register', None)
        unregister_func = getattr(register_types, 'unregister', None)
        terminate_func = getattr(register_types, 'terminate', None)   
    except ImportError:
        godot._print_rich("[color=orange]'register types' module was not found.[/color]")

    if init_func:
        # TODO: Call with init level, do all levels
        init_func()

    if register_func:
        register_func()


def terminate_types():
    global unregister_func, terminate_func
    # print("Python-level cleanup")

    if unregister_func:
        unregister_func()
    if terminate_func:
        # TODO: Call with init level
        terminate_func()
