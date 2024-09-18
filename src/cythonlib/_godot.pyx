cimport cython
from cpython cimport ref
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

import io
import sys
import pickle
import traceback

_print = print # print will be redefined for _godot module, keep the original

include "api_data.pxi"
include "shortcuts.pxi"

include "class.pxi"
include "object.pxi"
include "method_bind.pxi"

include "io.pxi"
include "register_types.pxi"
