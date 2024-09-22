from gdextension_interface cimport *

ctypedef float real_t

include "defs/variant.pxi"
include "defs/string.pxi"
include "defs/string_name.pxi"
include "defs/dictionary.pxi"
include "defs/array.pxi"

include "defs/vector2.pxi"

include "defs/classdb.pxi"
include "defs/engine.pxi"
include "defs/os.pxi"

include "defs/utility_functions.pxi"


# ctypedef fused type_t:
#     bint
#     uint32_t
#     real_t
#     String
#     StringName
#     Dictionary
#     Array
#     Vector2
