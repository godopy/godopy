cdef extern from "godot_cpp/variant/packed_color_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedColorArray:
        PackedColorArray()

        int64_t size() const
        bint is_empty() const

        const Color *ptr() const
        Color *ptrw()