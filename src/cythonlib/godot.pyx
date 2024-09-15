cimport cython
from cpython cimport ref
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

include "gde_shortcuts.pxi"
include "type_converters.pxi"

include "object.pxi"
include "method_bind.pxi"
include "class.pxi"

cdef GodotUtilityFunctionNoRet _printraw = GodotUtilityFunctionNoRet('printraw', 2648703342)
cdef GodotUtilityFunctionNoRet _print_rich = GodotUtilityFunctionNoRet('print_rich', 2648703342)
cdef GodotUtilityFunctionNoRet _push_error = GodotUtilityFunctionNoRet('push_error', 2648703342)
cdef GodotUtilityFunctionNoRet _push_warning = GodotUtilityFunctionNoRet('push_warning', 2648703342)

print = GodotUtilityFunctionNoRet('print', 2648703342)
printerr = GodotUtilityFunctionNoRet('printerr', 2648703342)
printt = GodotUtilityFunctionNoRet('printt', 2648703342)
prints = GodotUtilityFunctionNoRet('prints', 2648703342)
printverbose = GodotUtilityFunctionNoRet('print_verbose', 2648703342)

print_rich = _print_rich
printraw = _printraw
push_error = _push_error
push_warningg = _push_warning

_OS = GodotSingleton('OS')
_OS__read_string_from_stdin = GodotMethodBindRet(_OS, 'read_string_from_stdin', 2841200299, 'String')

cpdef input(str prompt=None):
    if prompt is not None:
        printraw(prompt)

    return _OS__read_string_from_stdin()

