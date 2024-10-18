cdef extern from "godot_cpp/variant/aabb.hpp" namespace "godot" nogil:
    cppclass _AABB "godot::AABB":
        Vector3 position
        Vector3 size

        _AABB()
        _AABB(const Vector3 &, const Vector3 &)
