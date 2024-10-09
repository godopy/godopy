cimport cython
from cpython cimport ref
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

import types
import pickle
import builtins
import traceback
import importlib

include "api_data.pxi"
include "shortcuts.pxi"
include "typeconv.pxi"

include "class.pxi"
include "object.pxi"
include "callable.pxi"
include "method_bind.pxi"
include "utility_function.pxi"

include "extension_virtual_method.pxi"
include "extension_method.pxi"
include "extension_class.pxi"
include "extension_callbacks.pxi"
include "extension_class_registrator.pxi"
include "extension.pxi"


cpdef input(str prompt=None):
    if prompt is not None:
        UtilityFunctions.printraw(prompt)

    return OS.get_singleton().read_string_from_stdin()


def get_class_method_list(str name, no_inheritance=False):
    cdef list data = (<Variant>ClassDB.get_singleton().class_get_method_list(name, no_inheritance)).pythonize()

    res = {}

    for item in data:
        name = item.pop('name')
        res[name] = item

    return res
