cdef extern from "godot_cpp/variant/rect2i.hpp" namespace "godot" nogil:
    cppclass Rect2i:
        Vector2i position
        Vector2i size

        Rect2i()
        Rect2i(const int32_t, const int32_t, const int32_t, const int32_t)
        Rect2i(const Vector2i &, const Vector2i &)
