cdef extern from "godot_cpp/variant/packed_vector3_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedVector3Array:
        PackedVector3Array()

        int64_t size() const
        bint is_empty() const

        int64_t resize(int64_t p_size)

        const Vector3 *ptr() const
        Vector3 *ptrw()
