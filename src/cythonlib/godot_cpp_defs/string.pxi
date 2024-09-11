cdef extern from "godot_cpp/variant/char_string.hpp" namespace "godot" nogil:
    cdef cppclass CharString:
        const char *get_data()

cdef extern from "godot_cpp/variant/string.hpp" namespace "godot" nogil:
    cppclass String:
        String()
        String(const char *)

        bint operator==(const String&)
        bint operator==(const char *)
        bint operator!=(const String&)

        CharString utf8()
