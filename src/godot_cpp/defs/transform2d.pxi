cdef extern from "godot_cpp/variant/transform2d.hpp" namespace "godot" nogil:
    cppclass Transform2D:
        Vector2[3] columns

        Transform2D()
        Transform2D(const real_t, const real_t, const real_t, const real_t, const real_t, const real_t)
        Transform2D(const Vector2 &, const Vector2 &)
