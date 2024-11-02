cdef extern from "godot_cpp/variant/vector4.hpp" namespace "godot" nogil:
    cdef cppclass Vector4:
        real_t x
        real_t y
        real_t z
        real_t w

        real_t[4] coord

        Vector4()
        Vector4(const real_t, const real_t, const real_t, const real_t)
