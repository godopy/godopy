cdef extern from "godot_cpp/variant/callable.hpp" namespace "godot" nogil:
    cdef cppclass GodotCppCallable "godot::Callable":
        GodotCppCallable()
        GodotCppCallable(const GodotCppCallable &)
        GodotCppCallable(GodotCppObject &, const StringName &)
        GodotCppCallable(GodotCppObject &, str)

        @staticmethod
        GodotCppCallable create(const Variant &, const StringName &)

        @staticmethod
        GodotCppCallable create(const Variant &, str)

        Variant callv(const Array &p_arguments) const
        bint is_null() const
        bint is_custom() const
        bint is_standard() const
        bint is_valid() const
        GodotCppObject *get_object() const
        int64_t get_object_id() const
        str get_method() const
        int64_t get_argument_count() const
        int64_t get_bound_arguments_count() const
        Array get_bound_arguments() const
        int64_t hash() const
        GodotCppCallable bindv(const Array &p_arguments)
        GodotCppCallable unbind(int64_t p_argcount) const

        Variant call(const Variant &)
        Variant call(const Variant &, const Variant &)
        Variant call(const Variant &, const Variant &, const Variant &)
        Variant call(const Variant &, const Variant &, const Variant &, const Variant &)
        Variant call(const Variant &, const Variant &, const Variant &, const Variant &, const Variant &)
