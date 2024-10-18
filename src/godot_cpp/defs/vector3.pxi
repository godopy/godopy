cdef extern from "godot_cpp/variant/vector3.hpp" namespace "godot" nogil:
    cppclass Vector3:
        real_t x
        real_t y
        real_t z

        real_t[3] coords

        Vector3()
        Vector3(const real_t, const real_t, const real_t)
