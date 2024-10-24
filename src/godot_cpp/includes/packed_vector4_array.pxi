cdef extern from "godot_cpp/variant/packed_vector4_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedVector4Array:
        PackedVector4Array()

        int64_t size() const
        bint is_empty() const

        const Vector4 *ptr() const
        Vector4 *ptrw()
