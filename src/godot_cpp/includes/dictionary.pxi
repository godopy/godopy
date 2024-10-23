cdef extern from "godot_cpp/variant/dictionary.hpp" namespace "godot" nogil:
    cdef cppclass Dictionary:
        Dictionary()

        int64_t size() const
        bint has(const Variant &p_key) const
        Array keys() const
        Array values() const
        Variant get(const Variant &p_key, const Variant &p_default) const
        bint is_typed() const

        void make_read_only()
        bint is_read_only() const

        const Variant &operator[](const Variant &p_key) const
        Variant &operator[](const Variant &p_key)
        # void set_typed(uint32_t p_key_type, const StringName &p_key_class_name, const Variant &p_key_script, uint32_t p_value_type, const StringName &p_value_class_name, const Variant &p_value_script)
