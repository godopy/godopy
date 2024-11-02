cdef extern from "godot_cpp/variant/vector2.hpp" namespace "godot" nogil:
    cdef cppclass Vector2:
        real_t x
        real_t y

        real_t width
        real_t height

        real_t[2] coord

        Vector2()
        Vector2(const real_t, const real_t)
