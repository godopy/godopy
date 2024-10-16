cdef extern from "godot_cpp/variant/callable.hpp" namespace "godot" nogil:
    cppclass _Callable "godot::Callable":
        _Callable()
        _Callable(const _Callable &)
        _Callable(_Object &, const StringName &)
        _Callable(_Object &, str)

        @staticmethod
        _Callable create(const Variant &, const StringName &)

        @staticmethod
        _Callable create(const Variant &, str)

        Variant callv(const Array &p_arguments) const
        bint is_null() const
        bint is_custom() const
        bint is_standard() const
        bint is_valid() const
        _Object *get_object() const
        int64_t get_object_id() const
        str get_method() const
        int64_t get_argument_count() const
        int64_t get_bound_arguments_count() const
        Array get_bound_arguments() const
        int64_t hash() const
        _Callable bindv(const Array &p_arguments)
        _Callable unbind(int64_t p_argcount) const

        Variant call(const Variant &)
        Variant call(const Variant &, const Variant &)
        Variant call(const Variant &, const Variant &, const Variant &)
        Variant call(const Variant &, const Variant &, const Variant &, const Variant &)
        Variant call(const Variant &, const Variant &, const Variant &, const Variant &, const Variant &)
