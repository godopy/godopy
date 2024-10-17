cdef extern from "godot_cpp/variant/rect2i.hpp" namespace "godot" nogil:
    cppclass Rect2i:
        Rect2i()
        Rect2i(int32_t, int32_t, int32_t, int32_t)
