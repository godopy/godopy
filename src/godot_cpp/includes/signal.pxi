cdef extern from "godot_cpp/variant/signal.hpp" namespace "godot" nogil:
    cdef cppclass GodotCppSignal "godot::Signal":
        GodotCppSignal()
        GodotCppSignal(const GodotCppSignal &)
        GodotCppSignal(GodotCppObject &, const StringName &)
        GodotCppSignal(GodotCppObject &, str)

        bint is_null() const

        GodotCppObject *get_object() const
        int64_t get_object_id() const
        str get_name() const

        int64_t connect(const GodotCppCallable &p_callable)
        int64_t connect(const GodotCppCallable &p_callable, int64_t)
        void disconnect(const GodotCppCallable &p_callable)
        bint is_connected(const GodotCppCallable &p_callable) const
        Array get_connections() const

        void emit(const Variant &)
        void emit(const Variant &, const Variant &, const Variant &)
        void emit(const Variant &, const Variant &, const Variant &, const Variant &)
        void emit(const Variant &, const Variant &, const Variant &, const Variant &, const Variant &)
