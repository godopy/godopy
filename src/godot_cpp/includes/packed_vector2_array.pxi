cdef extern from "godot_cpp/variant/packed_vector2_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedVector2Array:
        PackedVector2Array()

        int64_t size() const
        bint is_empty() const

        const Vector2 *ptr() const
        Vector2 *ptrw()
