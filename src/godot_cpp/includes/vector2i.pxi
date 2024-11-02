cdef extern from "godot_cpp/variant/vector2i.hpp" namespace "godot" nogil:
    cdef cppclass Vector2i:
        int32_t x
        int32_t y

        int32_t width
        int32_t height

        int32_t[2] coord

        Vector2i()
        Vector2i(const int32_t, const int32_t)
