cdef extern from "godot_cpp/variant/packed_int32_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedInt32Array:
        PackedInt32Array()

        int64_t size() const
        bint is_empty() const

        int64_t resize(int64_t p_size)

        const int32_t *ptr() const
        int32_t *ptrw()
