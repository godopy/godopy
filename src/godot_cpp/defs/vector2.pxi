cdef extern from "godot_cpp/variant/vector2.hpp" namespace "godot" nogil:
    cppclass Vector2:
        Vector2()
        Vector2(real_t, real_t)