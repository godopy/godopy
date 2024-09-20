
cdef class ExtensionMethod(ExtensionVirtualMethod):
    @staticmethod
    cdef void bind_call(
        void *p_method_userdata,
        GDExtensionClassInstancePtr p_instance,
        const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
        GDExtensionVariantPtr r_return,
        GDExtensionCallError *r_error
    ) noexcept nogil:
        cdef Variant ret
        with gil:
            ret = ExtensionMethod.bind_call_gil(p_method_userdata, p_instance,
                                                     p_args, p_argument_count, r_error)
        _gde_variant_new_copy(r_return, ret._native_ptr())

    @staticmethod
    cdef Variant bind_call_gil(
        void *p_method_userdata,
        GDExtensionClassInstancePtr p_instance,
        const GDExtensionConstVariantPtr *p_args,
        GDExtensionInt p_argument_count,
        GDExtensionCallError *r_error
    ):
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
    cdef void bind_ptrcall(
        void *p_method_userdata,
        GDExtensionClassInstancePtr p_instance,
        const GDExtensionConstTypePtr *p_args,
        GDExtensionTypePtr r_return
    ) noexcept nogil:
        with gil:
            ExtensionMethod.bind_ptrcall_gil(p_method_userdata, p_instance, p_args, r_return)

    @staticmethod
    cdef void bind_ptrcall_gil(
        void *p_method_userdata,
        GDExtensionClassInstancePtr p_instance,
        const GDExtensionConstTypePtr *p_args,
        GDExtensionTypePtr r_return
    ):
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


    cdef int register(self) except -1:
        cdef GDExtensionClassMethodInfo mi

        if self.get_argument_count() < 1:
            raise RuntimeError('At least 1 argument ("self") is required')

        cdef PropertyInfo _return_value_info = self.get_return_info()
        cdef GDExtensionPropertyInfo return_value_info

        return_value_info.type = <GDExtensionVariantType>_return_value_info.type
        return_value_info.name = SN(_return_value_info.name).ptr()
        return_value_info.class_name = SN(_return_value_info.class_name).ptr()
        return_value_info.hint = _return_value_info.hint
        return_value_info.hint_string = SN(_return_value_info.hint_string).ptr() 
        return_value_info.usage = _return_value_info.usage

        # print('RETURN: %s' % _return_value_info)

        cdef size_t i

        cdef list _def_args = self.get_default_arguments()
        cdef GDExtensionVariantPtr *def_args = <GDExtensionVariantPtr *> \
            _gde_mem_alloc(len(_def_args) * cython.sizeof(GDExtensionVariantPtr))
        cdef Variant defarg
        for i in range(len(_def_args)):
            defarg = <Variant>_def_args[i]
            def_args[i] = <GDExtensionVariantPtr>&defarg

        # Skip self arg
        cdef list _arguments_info = self.get_argument_info_list()[1:]
        cdef size_t argsize = len(_arguments_info)
        cdef GDExtensionPropertyInfo *arguments_info = <GDExtensionPropertyInfo *> \
            _gde_mem_alloc(argsize * cython.sizeof(GDExtensionPropertyInfo))

        cdef str pyname
        cdef str pyclassname
        cdef str pyhintstring
        cdef int pytype

        for i in range(argsize):
            pyname = _arguments_info[i].name
            pyclassname = _arguments_info[i].class_name
            pyhintstring = _arguments_info[i].hint_string
            pytype = _arguments_info[i].type
            arguments_info[i].type = <GDExtensionVariantType>pytype
            arguments_info[i].name = (SN(pyname)).ptr()
            arguments_info[i].class_name = (SN(pyclassname)).ptr()
            arguments_info[i].hint = _arguments_info[i].hint
            arguments_info[i].hint_string = (SN(pyhintstring)).ptr()
            arguments_info[i].usage = _arguments_info[i].usage

        # print('ARGS: %s' % _arguments_info)

        cdef list _arguments_metadata = self.get_argument_metadata_list()[1:]
        cdef int *arguments_metadata = <int *>_gde_mem_alloc(len(_arguments_metadata) * cython.sizeof(int))
        for i in range(len(_arguments_metadata)):
            arguments_metadata[i] = <int>_arguments_metadata[i]

        cdef GDExtensionClassMethodArgumentMetadata return_value_metadata = \
            <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()

        cdef str method_name = self.__name__
        cdef StringName _method_name = StringName(method_name)

        mi.name = _method_name._native_ptr()
        mi.method_userdata = <void *><PyObject *>self
        mi.call_func = &ExtensionMethod.bind_call
        mi.ptrcall_func = &ExtensionMethod.bind_ptrcall
        mi.method_flags = self.get_hint_flags()
        mi.has_return_value = self.has_return()
        mi.return_value_info = NULL # &return_value_info
        mi.return_value_metadata = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE
        mi.argument_count = argsize
        mi.arguments_info = arguments_info
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>arguments_metadata
        mi.default_argument_count = self.get_default_argument_count()
        mi.default_arguments = def_args

        ref.Py_INCREF(self)  # DECREF ??? TODO

        print("REG METHOD %s:%s %x" % (self.owner_class.__name__, self.__name__, <uint64_t><PyObject *>self))
        cdef str name = self.owner_class.__name__

        _gde_classdb_register_extension_class_method(gdextension_library, SN(name).ptr(), &mi)

        _gde_mem_free(def_args)
        _gde_mem_free(arguments_info)
        _gde_mem_free(arguments_metadata)

        return 0


cdef inline void set_variant_from_ptr(Variant *v, Variant value) noexcept nogil:
    cdef Variant result = cython.operator.dereference(v)
    result = value
