cimport cython
from cpython cimport ref
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

import sys
import types

include "gde_shortcuts.pxi"
include "type_converters.pxi"

include "object.pxi"
include "method_bind.pxi"
include "class.pxi"

cdef GodotUtilityFunctionNoRet printraw = GodotUtilityFunctionNoRet('printraw', 2648703342)
cdef GodotUtilityFunctionNoRet print_rich = GodotUtilityFunctionNoRet('print_rich', 2648703342)
cdef GodotUtilityFunctionNoRet push_error = GodotUtilityFunctionNoRet('push_error', 2648703342)

include "sys_util.pxi"
