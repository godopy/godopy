cdef extern from "godot_cpp/variant/packed_vector3_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedVector3Array:
        PackedVector3Array()

        int64_t size() const
        bint is_empty() const

        const Vector3 *ptr() const
        Vector3 *ptrw()
