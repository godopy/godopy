cdef extern from "godot_cpp/variant/plane.hpp" namespace "godot" nogil:
    cppclass Plane:
        Vector3 normal
        real_t d

        Plane()
        Plane(real_t, real_t, real_t, real_t)
        Plane(const Vector3 &, real_t)
        Plane(const Vector3 &, const Vector3 &)

