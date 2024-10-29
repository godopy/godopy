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

import numpy as np
import godot_types as gdtypes

include "api_data.pxi"

cdef set _global_singleton_info = pickle.loads(_global_singleton_info__pickle)
cdef dict _global_enum_info = pickle.loads(_global_enum_info__pickle)
# cdef dict _global_struct_info = pickle.loads(_global_struct_info__pickle)
cdef dict _global_inheritance_info = pickle.loads(_global_inheritance_info__pickle)
cdef dict _global_utility_function_info = pickle.loads(_global_utility_function_info__pickle)

# cdef set _global_core_classes = pickle.loads(_global_api_types__pickles['core'])
# cdef set _global_editor_classes = pickle.loads(_global_api_types__pickles['editor'])

include "includes/typeconv.pxi"
include "includes/exceptions.pxi"

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

include "includes/extension_class.pxi"
include "includes/extension_callbacks.pxi"
include "includes/extension_class_registrator.pxi"
include "includes/extension.pxi"


cpdef input(str prompt=None):
    if prompt is not None:
        UtilityFunctions.printraw(prompt)

    return OS.get_singleton().read_string_from_stdin()


def _has_class(str name):
    return name in _global_inheritance_info

def _has_singleton(str name):
    return name in _global_singleton_info


def _classdb_dir():
    return _global_inheritance_info.keys()

def _singletons_dir():
    return  _global_singleton_info

def _utility_functions_dir():
    return _global_utility_function_info

def _enums_dir():
    return _global_enum_info

# def _structs_dir():
#     return _global_struct_info

def _get_class_method_list(str name, no_inheritance=False):
    cdef list data = (<Variant>ClassDB.get_singleton().class_get_method_list(name, no_inheritance)).pythonize()

    res = {}

    for item in data:
        name = item.pop('name')
        res[name] = item

    return res
