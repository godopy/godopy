cdef extern from "godot_cpp/variant/packed_string_array.hpp" namespace "godot" nogil:
    cdef cppclass PackedStringArray:
        PackedStringArray()
        PackedStringArray(object)

        int64_t size() const
        bint is_empty() const

        void set(int64_t p_index, const String &p_value)
        void push_back(const String &)
        bint has(const String &p_value) const

        int64_t resize(int64_t p_new_size)

        const String &operator[](int64_t p_index) const
        String &operator[](int64_t p_index)

        const String *ptr() const
        String *ptrw()

