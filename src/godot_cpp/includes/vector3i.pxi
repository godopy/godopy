cdef extern from "godot_cpp/variant/vector3i.hpp" namespace "godot" nogil:
    cdef cppclass Vector3i:
        int32_t x
        int32_t y
        int32_t z

        int32_t[3] coord

        Vector3i()
        Vector3i(const int32_t, const int32_t, const int32_t)
