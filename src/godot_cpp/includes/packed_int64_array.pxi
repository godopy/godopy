cdef extern from "godot_cpp/variant/packed_int64_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedInt64Array:
        PackedInt64Array()

        int64_t size() const
        bint is_empty() const

        int64_t resize(int64_t p_size)

        const int64_t *ptr() const
        int64_t *ptrw()
