"""\
This module wraps GodoPy extensions to the engine
and holds the extension runtime data
"""
cimport cython
from cpython cimport ref, PyObject
from libcpp.vector cimport vector
from godot_cpp cimport *
from cython.operator cimport dereference as deref

cimport _godot as gd

import types

include "shortcuts.pxi"
include "typeconv.pxi"

include "extension_virtual_method.pxi"
include "extension_method.pxi"
include "extension_class.pxi"
include "extension_callbacks.pxi"
include "extension_class_registrator.pxi"
include "extension.pxi"
