cdef class ExtensionMethod:
    cdef ExtensionClass owner_class
    cdef object method
    cdef str __name__

    @staticmethod
    cdef void bind_call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                        const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
                        GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) noexcept nogil:
        cdef Variant ret
        with gil:
            ret = ExtensionMethod.bind_call_gil(p_method_userdata, p_instance,
                                                     p_args, p_argument_count, r_error)
        _gde_variant_new_copy(r_return, ret._native_ptr())

    @staticmethod
    cdef Variant bind_call_gil(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                               const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
                               GDExtensionCallError *r_error):
        cdef ExtensionMethod self = <object>p_method_userdata
        cdef gd.Object wrapper = <object>p_instance
        cdef int i
        cdef list args = []
        cdef Variant arg
        for i in range(p_argument_count):
            arg = deref(<Variant *>p_args[i])
            args.append(<object>arg)
        cdef object ret = self.method(wrapper, *args)
        return <Variant>ret

    @staticmethod
    cdef void bind_ptrcall(void *p_method_userdata,
                           GDExtensionClassInstancePtr p_instance,
                           const GDExtensionConstTypePtr *p_args,
                           GDExtensionTypePtr r_return) noexcept nogil:
        with gil:
            ExtensionMethod.bind_ptrcall_gil(p_method_userdata, p_instance, p_args, r_return)

    @staticmethod
    cdef void bind_ptrcall_gil(void *p_method_userdata,
                               GDExtensionClassInstancePtr p_instance,
                               const GDExtensionConstTypePtr *p_args,
                               GDExtensionTypePtr r_return):
        cdef ExtensionMethod self = <object>p_method_userdata
        cdef gd.Object wrapper = <object>p_instance
        cdef size_t i = 0
        cdef list args = []
        cdef Variant arg
        while p_args[i] != NULL:
            arg = deref(<Variant *>p_args[i])
            args.append(<object>arg)
            i += 1
        cdef object ret = self.method(wrapper, *args)
        cdef Variant gd_ret = <Variant>ret
        set_variant_from_ptr(<Variant *>r_return, gd_ret)

    cdef GDExtensionVariantPtr *get_default_arguments(self):
        cdef vector[GDExtensionVariantPtr] def_args
        def_args.resize(len(self.method.__defaults__))
        cdef Variant arg
        for i in range(len(self.method.__defaults__)):
            arg = <Variant>self.method.__defaults__[i]
            def_args[i] = <GDExtensionVariantPtr>&arg

        return def_args.data()

    cdef GDExtensionPropertyInfo get_argument_info(self, int pos):
        cdef GDExtensionPropertyInfo pi

        pi.type = GDEXTENSION_VARIANT_TYPE_NIL  # Get from func.__anotations__ if possible
        pi.name = NULL
        cdef str class_name = self.method.owner_class.__name__
        pi.class_name = StringName(class_name)._native_ptr()
        pi.hint = 0
        pi.hint_string = NULL
        pi.usage = 0

        cdef object varname = None
        cdef StringName _name
        if pos >= 0:
            try:
                varname = self.method.__code__.co_varnames[pos]
                pi.name = StringName(<str>varname)._native_ptr()
            except IndexError:
                pass

        return pi

    cdef GDExtensionPropertyInfo *get_argument_info_list(self):
        cdef vector[GDExtensionPropertyInfo] arg_info

        cdef int i
        cdef argcount = self.argument_count()
        arg_info.reserve(argcount + 1) # 0 is return value
        for i in range(argcount + 1):
            arg_info.push_back(self.get_argument_info(i - 1))

        return arg_info.data()

    cdef GDExtensionClassMethodArgumentMetadata *get_argument_metadata_list(self):
        cdef vector[GDExtensionClassMethodArgumentMetadata] meta_info

        cdef int i
        cdef argcount = self.argument_count()
        meta_info.reserve(argcount + 1) # 0 is return value
        for i in range(argcount + 1):
            meta_info.push_back(GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE)

        return meta_info.data()

    cdef GDExtensionBool has_return(self):
        return <GDExtensionBool>bool(self.method.annotations.get('return'))

    cdef uint32_t get_hint_flags(self):
        return 0

    cdef uint32_t get_argument_count(self):
        return <uint32_t>self.method.__code__.co_argcount

    cdef uint32_t get_default_argument_count(self):
        return <uint32_t>len(self.method.__defaults__)

    def __init__(self, ExtensionClass owner_class, object method: types.FunctionType):
        self.owner_class = owner_class
        self.method = method
        self.__name__ = method.__name__


cdef inline void set_variant_from_ptr(Variant *v, Variant value) noexcept nogil:
    cdef Variant result = cython.operator.dereference(v)
    result = value
