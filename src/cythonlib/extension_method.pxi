cdef class PropertyInfo:
    cdef public int type
    cdef public str name
    cdef public str class_name
    cdef public uint32_t hint
    cdef public str hint_string
    cdef public uint32_t usage

    def __cinit__(self, int type, str name, str class_name, uint32_t hint=0, str hint_string='', uint32_t usage=0):
        self.type = type
        self.name = name
        self.class_name = class_name
        self.hint = hint
        self.hint_string = hint_string
        self.usage = usage

    def __repr__(self):
        return '<PropertyInfo %s:%s:%s>' % (self.class_name, self.name, vartype_to_str(self.type))

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
        print("METHOD CALL %x" % <uint64_t>p_instance)
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
        print("METHOD PTRCALL %x" % <uint64_t>p_instance)
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

    cdef list get_default_arguments(self):
        if self.method.__defaults__ is None:
            return []
        return [arg for arg in self.method.__defaults__]

    cdef PropertyInfo get_argument_info(self, int pos):
        cdef PropertyInfo pi = PropertyInfo(
            <int>GDEXTENSION_VARIANT_TYPE_FLOAT,
            '',
            self.owner_class.__name__
        )
        if pos >= 0:
            try:
                pi.name = self.method.__code__.co_varnames[pos]
            except IndexError:
                gd._push_error('Argname is missing in method %s, pos %d' % (self.method.__name__, pos))

        return pi

    cdef PropertyInfo get_return_info(self):
        return PropertyInfo(
            <int>GDEXTENSION_VARIANT_TYPE_NIL,
            '',
            self.owner_class.__name__
        )

    cdef list get_argument_info_list(self):
        return [self.get_argument_info(i) for i in range(self.get_argument_count())]


    cdef int get_return_metadata(self):
        return <int>GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE

    cdef list get_argument_metadata_list(self):
        cdef size_t i
        return [<int>GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE for i in range(self.get_argument_count())]

    cdef GDExtensionBool has_return(self):
        return <GDExtensionBool>bool(self.method.__annotations__.get('return'))

    cdef uint32_t get_hint_flags(self):
        return 0

    cdef uint32_t get_argument_count(self):
        return <uint32_t>self.method.__code__.co_argcount

    cdef uint32_t get_default_argument_count(self):
        if self.method.__defaults__ is None:
            return 0
        return <uint32_t>len(self.method.__defaults__)

    def __init__(self, ExtensionClass owner_class, object method: types.FunctionType):
        self.owner_class = owner_class
        self.method = method
        self.__name__ = method.__name__


cdef inline void set_variant_from_ptr(Variant *v, Variant value) noexcept nogil:
    cdef Variant result = cython.operator.dereference(v)
    result = value
