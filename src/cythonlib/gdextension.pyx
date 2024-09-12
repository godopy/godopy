cimport cython
from cpython cimport ref, PyObject
from libcpp.vector cimport vector
from godot_cpp cimport *
from cython.operator cimport dereference as deref

cimport godot

import types
import builtins

include "gde_shortcuts.pxi"
include "type_converters_ext.pxi"

include "method.pxi"
include "extension.pxi"
include "extension_class.pxi"

def hello():
    print("Hello from gdextension module!")
