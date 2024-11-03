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
from collections import namedtuple

import numpy as np
import godot_types as gdtypes

include "api_data.pxi"

cdef set _global_singleton_info = pickle.loads(_global_singleton_info__pickle)
cdef dict _global_enum_info = pickle.loads(_global_enum_info__pickle)
cdef dict _global_struct_info = pickle.loads(_global_struct_info__pickle)
cdef dict _global_inheritance_info = pickle.loads(_global_inheritance_info__pickle)
cdef dict _global_utility_function_info = pickle.loads(_global_utility_function_info__pickle)

# Custom class defined in C++
_global_inheritance_info['PythonObject'] = 'RefCounted'


cdef int configure(object config) except -1:
    config['object_constructor'] = Object
    config['godot_class_to_class'] = lambda _: Extension


include "includes/typeconv.pxi"
include "includes/exceptions.pxi"


### INTERFACE

GodotVersion = namedtuple('GodotVersion', ['major', 'minor', 'pach', 'string'])

def get_godot_version():
    """
    Gets the Godot version that the GDExtension was loaded into.
    """
    cdef GDExtensionGodotVersion version
    gdextension_interface_get_godot_version(&version)

    return GodotVersion(version.major, version.minor, version.patch, version.string)


### INTERFACE: Memory

cdef class _Memory:
    cdef void *ptr
    cdef size_t num_bytes

    def __cinit__(self, size_t p_bytes):
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

    def __dealloc__(self):
        self.free()


### INTERFACE: Godot Core






### INTERFACE: Object/ClassDB
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

### INTERFACE: ClassDB Extension
include "includes/extension_class.pxi"
include "includes/extension_callbacks.pxi"
include "includes/extension_class_registrator.pxi"
include "includes/extension.pxi"


cpdef input(object prompt=None):
    if prompt is not None:
        UtilityFunctions.printraw(prompt)

    cdef String s = OS.get_singleton().read_string_from_stdin()

    return type_funcs.string_to_pyobject(s)


def _has_class(name):
    return name in _global_inheritance_info

def _has_singleton(name):
    return name in _global_singleton_info


def _classdb_dir():
    return _global_inheritance_info.keys()

def _singletons_dir():
    return  _global_singleton_info

def _utility_functions_dir():
    return _global_utility_function_info

def _enums_dir():
    return _global_enum_info
