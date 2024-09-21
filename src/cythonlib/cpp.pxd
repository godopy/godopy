from gdextension_interface cimport *

ctypedef float real_t

include "_cpp/godot.pxi"

include "_cpp/variant.pxi"
include "_cpp/string.pxi"
include "_cpp/string_name.pxi"
include "_cpp/dictionary.pxi"
include "_cpp/array.pxi"

include "_cpp/vector2.pxi"

include "_cpp/classdb.pxi"
include "_cpp/engine.pxi"
include "_cpp/os.pxi"

include "_cpp/utility_functions.pxi"


# ctypedef fused type_t:
#     bint
#     uint32_t
#     real_t
#     String
#     StringName
#     Dictionary
#     Array
#     Vector2
