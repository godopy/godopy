cdef extern from "godot_cpp/variant/transform3d.hpp" namespace "godot" nogil:
    cppclass Transform3D:
        Basis basis
        Vector3 origin

        Transform3D()
        Transform3D(const Basis &, const Vector3 &)
        Transform3D(const Vector3 &, const Vector3 &, const Vector3 &, const Vector3 &)
        Transform3D(real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t)
