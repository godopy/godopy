cdef extern from "godot_cpp/variant/signal.hpp" namespace "godot" nogil:
    cppclass _Signal "godot::Signal":
        _Signal()
        _Signal(const _Signal &)
        _Signal(_Object &, const StringName &)
        _Signal(_Object &, str)

        bint is_null() const

        _Object *get_object() const
        int64_t get_object_id() const
        str get_name() const

        int64_t connect(const _Callable &p_callable)
        int64_t connect(const _Callable &p_callable, int64_t)
        void disconnect(const _Callable &p_callable)
        bint is_connected(const _Callable &p_callable) const
        Array get_connections() const

        void emit(const Variant &)
        void emit(const Variant &, const Variant &, const Variant &)
        void emit(const Variant &, const Variant &, const Variant &, const Variant &)
        void emit(const Variant &, const Variant &, const Variant &, const Variant &, const Variant &)
