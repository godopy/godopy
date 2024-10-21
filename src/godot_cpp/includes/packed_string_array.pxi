cdef extern from "godot_cpp/variant/packed_string_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedStringArray:
        PackedStringArray()
        PackedStringArray(object)

        void push_back(const String &)
