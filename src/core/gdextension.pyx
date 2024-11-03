cimport cython
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cpython cimport (
    PyObject, ref, pystate, PyLong_Check, PyLong_AsSsize_t,
    PyList_New, PyList_SET_ITEM, PyTuple_New, PyTuple_SET_ITEM
)

from python_runtime cimport *
cimport godot_types as type_funcs
from godot_types cimport StringName as PyStringName, variant_to_pyobject_func_t, variant_from_pyobject_func_t

import sys
import types
import pickle
import builtins
import traceback
import importlib
from typing import Dict, List, Set, Tuple
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
include "includes/exceptions.pxi"


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

cdef class _Memory:
    """
    Allocates, reallocates and frees memory.
    """
    cdef void *ptr
    cdef size_t num_bytes

    def __cinit__(self, size_t p_bytes) -> None:
        "Allocates memory."

        self.ptr = gdextension_interface_mem_alloc(p_bytes)
        self.num_bytes = p_bytes

        if self.ptr == NULL:
            raise MemoryError()

    cdef void *realloc(self, size_t p_bytes) except NULL nogil:
        "Reallocates memory."

        self.ptr = gdextension_interface_mem_realloc(self.ptr, p_bytes)

        if self.ptr == NULL:
            with gil:
                raise MemoryError()

        self.num_bytes = p_bytes

        return self.ptr

    cdef void free(self) noexcept nogil:
        "Frees memory."

        if self.ptr != NULL:
            gdextension_interface_mem_free(self.ptr)

            self.ptr = NULL

    def __dealloc__(self) -> None:
        self.free()


# INTERFACE: Godot Core

cpdef print_error(description, function, file, int32_t line, bint editor_notify):
    """
    Logs an error to Godot's built-in debugger and to the OS terminal.
    """
    gdextension_interface_print_error(description, function, file, line, editor_notify)

cpdef print_error_with_message(description, message, function, file, int32_t line, bint editor_notify):
    """
    Logs an error with a message to Godot's built-in debugger and to the OS terminal.
    """
    gdextension_interface_print_error_with_message(description, message, function, file, line, editor_notify)


cpdef print_warning(description, function, file, int32_t line, bint editor_notify):
    """
    Logs a warning to Godot's built-in debugger and to the OS terminal.
    """
    gdextension_interface_print_warning(description, function, file, line, editor_notify)

cpdef print_warning_with_message(description, message, function, file, int32_t line, bint editor_notify):
    """
    Logs a warning with a message to Godot's built-in debugger and to the OS terminal.
    """
    gdextension_interface_print_warning_with_message(description, message, function, file, line, editor_notify)


cpdef print_script_error(description, function, file, int32_t line, bint editor_notify):
    """
    Logs a script error to Godot's built-in debugger and to the OS terminal.
    """
    gdextension_interface_print_script_error(description, function, file, line, editor_notify)

cpdef print_script_error_with_message(description, message, function, file, int32_t line, bint editor_notify):
    """
    Logs a script error with a message to Godot's built-in debugger and to the OS terminal.
    """
    gdextension_interface_print_script_error_with_message(description, message, function, file, line, editor_notify)


# INTERFACE: ClassDB/Object

include "includes/class.pxi"
include "includes/object.pxi"

include "includes/engine_calls.pxi"
include "includes/engine_callable.pxi"
include "includes/method_bind.pxi"
include "includes/utility_function.pxi"
include "includes/builtin_method.pxi"

include "includes/python_calls.pxi"
include "includes/python_callable.pxi"
include "includes/extension_method_base.pxi"
include "includes/extension_virtual_method.pxi"
include "includes/extension_method.pxi"

# INTERFACE: ClassDB Extension

include "includes/extension_class.pxi"
include "includes/extension_callbacks.pxi"
include "includes/extension_class_registrator.pxi"
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
