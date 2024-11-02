cdef extern from "godot_cpp/variant/packed_byte_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedByteArray:
        PackedByteArray()

        int64_t size() const
        bint is_empty() const

        int64_t resize(int64_t p_size)

        const uint8_t *ptr() const
        uint8_t *ptrw()
