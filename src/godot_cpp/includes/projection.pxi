cdef extern from "godot_cpp/variant/projection.hpp" namespace "godot" nogil:
    cppclass Projection:
        Vector4[4] columns

        Projection()
        Projection(const Vector4 &, const Vector4 &, const Vector4 &, const Vector4 &)
        Projection(Transform3D &)
