cdef extern from "godot_cpp/variant/vector2i.hpp" namespace "godot" nogil:
    cppclass Vector2i:
        Vector2i()
        Vector2i(int32_t, int32_t)
