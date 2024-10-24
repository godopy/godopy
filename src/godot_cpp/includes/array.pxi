cdef extern from "godot_cpp/variant/array.hpp" namespace "godot" nogil:
    cdef cppclass Array:
        Array()

        int64_t size() const
        void push_back(const Variant &p_value)
        int64_t resize(int64_t p_size)

        Array map(const GodotCppCallable &p_method) const

        bint is_typed() const
        bint is_same_typed(const Array &p_array) const
        int64_t get_typed_builtin() const

        const Variant &operator[](int64_t p_index) const
        Variant &operator[](int64_t p_index)

        void set_typed(int vartype, const StringName &, const Variant &)
        void _ref(const Array &p_from) const


cdef extern from "godot_cpp/variant/typed_array.hpp" namespace "godot" nogil:
    cdef cppclass TypedArray[T]:
        TypedArray()
        TypedArray(const Array &)

        int64_t size() const
        void push_back(const Variant &p_value)
        int64_t resize(int64_t p_size)

        TypedArray map(const GodotCppCallable &p_method) const

    cdef cppclass TypedArrayBool "godot::TypedArray<bool>":
        TypedArrayBool()

        int64_t size() const
        int64_t resize(int64_t p_size)
