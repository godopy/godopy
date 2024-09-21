cimport cython
from cpython cimport ref
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

import io
import sys
import pickle
import builtins
import traceback

_print = print # print will be redefined for _godot module, keep the original

include "api_data.pxi"
include "shortcuts.pxi"
include "typeconv.pxi"

include "class.pxi"
include "object.pxi"
include "callable.pxi"
include "method_bind.pxi"
include "utility_function.pxi"

include "io.pxi"
include "register_types.pxi"


def get_class_method_list(str name, no_inheritance=False):
    cdef list data = (<Variant>ClassDB.get_singleton().class_get_method_list(name, no_inheritance)).pythonize()

    res = {}

    for item in data:
        name = item.pop('name')
        res[name] = item

    return res
