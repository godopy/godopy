"""\
This module provides a reasonably low-level Python implementation
of the GDExtension API.
"""

cimport cython
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cpython cimport (
    PyObject, ref, pystate, PyLong_Check, PyLong_AsSsize_t,
    PyList_New, PyList_SET_ITEM, PyTuple_New, PyTuple_SET_ITEM
)
cdef extern from "Python.h":
    Py_ssize_t Py_REFCNT(object o)
    Py_ssize_t _Py_REFCNT "Py_REFCNT" (PyObject *ptr)


from python_runtime cimport *
cimport godot_types as type_funcs
from godot_types cimport (
    StringName as PyStringName,
    variant_to_pyobject_func_t, variant_from_pyobject_func_t
)
from _gdextension_internals cimport print_traceback_and_die

import io
import sys
import types
import pickle
import inspect
import builtins
import traceback
import typing
from typing import Any, AnyStr, Dict, List, Optional, Set, Sequence, Tuple
from collections import namedtuple

import numpy as np
import godot_types as gdtypes

include "api_data.pxi"

cdef set _global_singleton_info = pickle.loads(_global_singleton_info__pickle)
cdef dict _global_enum_info = pickle.loads(_global_enum_info__pickle)
cdef set _global_struct_info = pickle.loads(_global_struct_info__pickle)
cdef dict _global_inheritance_info = pickle.loads(_global_inheritance_info__pickle)
cdef dict _global_utility_function_info = pickle.loads(_global_utility_function_info__pickle)

# Custom class defined in C++
_global_inheritance_info['PythonObject'] = 'RefCounted'


cdef int configure(object config) except -1:
    config['object_constructor'] = Object
    config['godot_class_to_class'] = lambda _: Extension


include "includes/typeconv.pxi"
include "includes/constants.pxi"


# INTERFACE

GodotVersion = namedtuple('GodotVersion', ['major', 'minor', 'pach', 'string'])

def get_godot_version() -> GodotVersion:
    """
    Gets the Godot version that the GDExtension was loaded into.
    """
    cdef GDExtensionGodotVersion version
    gdextension_interface_get_godot_version(&version)

    cdef str version_string = bytes(version.string).decode('utf-8')

    return GodotVersion(version.major, version.minor, version.patch, version_string)


def get_extension_api_version() -> GodotVersion:
    """
    Gets the Godot version from 'extension_api.json' that the GDExtension was built against.
    """
    return GodotVersion(
        _api_header['version_major'],
        _api_header['version_minor'],
        _api_header['version_patch'],
        _api_header['version_full_name']
    )


# INTERFACE: Memory

include "includes/memory.pxi"


# INTERFACE: Godot Core

include "includes/exceptions.pxi"


# INTERFACE: Object/ClassDB/Other

include "includes/class.pxi"
include "includes/object.pxi"

include "includes/engine_calls.pxi"
include "includes/method_bind.pxi"
include "includes/script_method.pxi"
include "includes/builtin_method.pxi"


# INTERFACE: Variant

include "includes/variant_method.pxi"
include "includes/variant_static_method.pxi"
include "includes/utility_function.pxi"


# INTERFACE: Script Instance

include "includes/method_info.pxi"
include "includes/python_calls.pxi"
include "includes/script_instance.pxi"


# INTERFACE: Callable

include "includes/callable.pxi"
include "includes/python_callable.pxi"


# INTERFACE: ClassDB Extension

include "includes/extension_method_base.pxi"
include "includes/extension_virtual_method.pxi"
include "includes/extension_method.pxi"

include "includes/extension_class.pxi"
include "includes/extension.pxi"


# Helpers

def has_class(name) -> bool:
    """
    Checks if a name is a class name in the Extension API or if it is a name of
    a custom GodoPy C++ GDExtension class.
    """
    return name in _global_inheritance_info

def has_singleton(name) -> bool:
    """
    Checks if a name is a singleton class name in the Extension API.
    """
    return name in _global_singleton_info


def classdb_set() -> Set[str]:
    """
    Returns a set of all class names in the Extension API and custom GodoPy C++ GDExtension classes.
    """
    return set(_global_inheritance_info.keys())

def singletons_set() -> Set[str]:
    """
    Returns a set of all singleton class names in the Extension API.
    """
    return _global_singleton_info

def utility_functions_set() -> Set[str]:
    """
    Returns a set of all utility function names in the Extension API.
    """
    return set(_global_utility_function_info)

def global_enums_dict() -> Dict[str, List[Tuple[str, int]]]:
    """
    Returns a dict of all global enumerations in the Extension API.
    """
    return _global_enum_info
