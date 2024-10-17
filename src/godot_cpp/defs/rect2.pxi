cdef extern from "godot_cpp/variant/rect2.hpp" namespace "godot" nogil:
    cppclass Rect2:
        Rect2()
        Rect2(real_t, real_t, real_t, real_t)
