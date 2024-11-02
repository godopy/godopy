cdef extern from "godot_cpp/variant/packed_float64_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedFloat64Array:
        PackedFloat64Array()

        int64_t size() const
        bint is_empty() const

        int64_t resize(int64_t p_size)

        const double *ptr() const
        double *ptrw()
