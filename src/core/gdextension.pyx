"""\
This module provides a reasonably low-level Python implementation
of the GDExtension API.
"""

cimport cython
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cpython cimport (
    PyObject, ref, pystate, PyLong_Check, PyLong_AsSsize_t, PyBytes_AsString,
    PyList_New, PyList_SET_ITEM, PyTuple_New, PyTuple_SET_ITEM
)
cdef extern from "Python.h":
    Py_ssize_t Py_REFCNT(object o)
    Py_ssize_t _Py_REFCNT "Py_REFCNT" (PyObject *ptr)


from python_runtime cimport *
cimport godot_types as type_funcs
from godot_types cimport (
    variant_to_pyobject_func_t, variant_from_pyobject_func_t
)
from _gdextension_internals cimport print_traceback_and_die

import io
import sys
import enum
import types
import pickle
import typing
import inspect
import builtins
import traceback

from typing import Any, Dict, List, Optional, Set, Sequence, Tuple
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


# INTERFACE: Version

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


# INTERFACE: Errors & Warnings (Godot Core)

include "includes/exceptions.pxi"


# INTERFACE: Object

include "includes/object.pxi"
include "includes/engine_calls.pxi"
include "includes/script_method.pxi"


# INTERFACE: Variant

include "includes/variant_method.pxi"
include "includes/variant_static_method.pxi"
include "includes/builtin_method.pxi"
include "includes/utility_function.pxi"


# INTERFACE: Script Instance

include "includes/method_info.pxi"
include "includes/python_calls.pxi"
include "includes/script_instance.pxi"


# INTERFACE: Callable

include "includes/callable.pxi"
include "includes/python_callable.pxi"


# INTERFACE: ClassDB

include "includes/class.pxi"
include "includes/method_bind.pxi"


# INTERFACE: ClassDB Extension

include "includes/extension_method_base.pxi"
include "includes/extension_virtual_method.pxi"
include "includes/extension_method.pxi"
include "includes/extension_enum.pxi"
include "includes/extension_property.pxi"
include "includes/extension_signal.pxi"
include "includes/extension_class.pxi"
include "includes/extension.pxi"


# INTEFACE: Path

def get_library_path() -> str:
    """
    Gets the path to the current GDExtension library.
    """
    cdef String ret

    gdextension_interface_get_library_path(gdextension_library, ret._native_ptr())

    return type_funcs.string_to_pyobject(ret)


# INTERFACE: Editor Plugins

cdef class Editor:
    @staticmethod
    def add_plugin(p_class_name: Str) -> None:
        """
        Adds an editor plugin.
        """
        cdef PyGDStringName class_name = PyGDStringName(p_class_name)
        gdextension_interface_editor_add_plugin(class_name.ptr())

    @staticmethod
    def remove_plugin(p_class_name: Str) -> None:
        """
        Removes an editor plugin.
        """
        cdef PyGDStringName class_name = PyGDStringName(p_class_name)
        gdextension_interface_editor_remove_plugin(class_name.ptr())


# INTERFACE: Editor Help

cdef class EditorHelp:
    @staticmethod
    def load_xml(xml_data: Str | bytes) -> None:
        """
        Loads new XML-formatted documentation data in the editor.
        """
        cdef const char *c_str
        cdef size_t size
        cdef bytes data

        if isinstance(xml_data, str):
            data = xml_data.encode('utf-8')
        elif isinstance(xml_data, bytes):
            data = xml_data
        else:
            raise ValueError("Expected str or bytes, got %r" % type(xml_data))

        c_str = PyBytes_AsString(data)
        size = len(data)

        gdextension_interface_editor_help_load_xml_from_utf8_chars_and_len(c_str, size)


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
