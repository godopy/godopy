cdef extern from "godot_cpp/variant/vector3i.hpp" namespace "godot" nogil:
    cppclass Vector3i:
        int32_t x
        int32_t y
        int32_t z

        int32_t[3] coords

        Vector3i()
        Vector3i(const int32_t, const int32_t, const int32_t)
