cdef extern from "godot_cpp/variant/rect2.hpp" namespace "godot" nogil:

    cppclass Rect2:
        Vector2 position
        Vector2 size

        Rect2()
        Rect2(const real_t, const real_t, const real_t, const real_t)
        Rect2(const Vector2 &, const Vector2 &)
