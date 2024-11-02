cdef extern from "godot_cpp/variant/basis.hpp" namespace "godot" nogil:
    cdef cppclass Basis:
        Vector3[3] rows

        Basis()
        Basis(const Vector3 &, const Vector3 &)
        Basis(const Vector3 &, real_t, const Vector3 &)
        Basis(const Vector3 &, real_t)
        Basis(const Quaternion &, const Vector3 &)
        Basis(const Quaternion &)
