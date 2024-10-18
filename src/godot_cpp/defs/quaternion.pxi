cdef extern from "godot_cpp/variant/quaternion.hpp" namespace "godot" nogil:
    cppclass Quaternion:
        real_t x
        real_t y
        real_t z
        real_t w

        real_t[4] components

        Quaternion()
        Quaternion(real_t, real_t, real_t, real_t)
        Quaternion(const Vector3 &, real_t)
        Quaternion(const Vector3 &)
