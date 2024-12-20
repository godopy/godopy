cdef extern from "godot_cpp/variant/packed_float32_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedFloat32Array:
        PackedFloat32Array()

        int64_t size() const
        bint is_empty() const

        int64_t resize(int64_t p_size)

        const float *ptr() const
        float *ptrw()
