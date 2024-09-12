cimport cython
from cpython cimport ref, PyObject
from libcpp.vector cimport vector
from godot cimport *
from cython.operator cimport dereference as deref

import sys
import types
import builtins

include "gde_shortcuts.pxi"
include "type_converters.pxi"

include "object.pxi"
include "method_bind.pxi"

printraw = GodotUtilityFunctionNoRet('printraw', 2648703342)
print_rich = GodotUtilityFunctionNoRet('print_rich', 2648703342)
push_error = GodotUtilityFunctionNoRet('push_error', 2648703342)

include "class.pxi"
include "method.pxi"

include "sys_util.pxi"
