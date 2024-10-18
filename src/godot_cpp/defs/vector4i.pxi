cdef extern from "godot_cpp/variant/vector4i.hpp" namespace "godot" nogil:
    cppclass Vector4i:
        int32_t x
        int32_t y
        int32_t z
        int32_t w

        int32_t[4] coords

        Vector4i()
        Vector4i(const int32_t, const int32_t, const int32_t, const int32_t)
