cdef extern from "godot_cpp/templates/hashfuncs.hpp" namespace "godot" nogil:
    cdef uint32_t hash_djb2(const char *p_cstr)
    cdef uint32_t hash_djb2_buffer(const uint8_t *p_buff, int p_len)
    cdef uint32_t hash_djb2_buffer(const uint8_t *p_buff, int p_len, uint32_t p_prev)
    cdef uint32_t hash_djb2_one_32(uint32_t p_in)
    cdef uint32_t hash_djb2_one_32(uint32_t p_in, uint32_t p_prev)
    cdef uint32_t hash_one_uint64(const uint64_t p_int)

    cdef uint32_t hash_murmur3_one_32(uint32_t p_in)
    cdef uint32_t hash_murmur3_one_32(uint32_t p_in, uint32_t p_seed)

    cdef uint32_t hash_murmur3_one_float(float p_in)
    cdef uint32_t hash_murmur3_one_float(float p_in, uint32_t p_seed)

    cdef uint32_t hash_murmur3_one_64(uint64_t p_in)
    cdef uint32_t hash_murmur3_one_64(uint64_t p_in, uint32_t p_seed)

    cdef uint32_t hash_murmur3_one_double(double p_in)
    cdef uint32_t hash_murmur3_one_double(double p_in, uint32_t p_seed)

    cdef uint32_t hash_murmur3_buffer(const void *key, int length)
    cdef uint32_t hash_murmur3_buffer(const void *key, int length, const uint32_t seed)

    cdef uint32_t hash_djb2_one_float(double p_in)
    cdef uint32_t hash_djb2_one_float(double p_in, uint32_t p_prev)

    cdef uint64_t hash_djb2_one_float_64(double p_in)
    cdef uint64_t hash_djb2_one_float_64(double p_in, uint64_t p_prev)

    cdef uint64_t hash_djb2_one_64(uint64_t p_in)
    cdef uint64_t hash_djb2_one_64(uint64_t p_in, uint64_t p_prev)
