cdef class PyStringName:
    cdef str pyname
    cdef StringName name 

    def __cinit__(self, str name):
        self.pyname = name
        self.name = StringName(name)


    cdef void *ptr(self):
        return self.name._native_ptr()


cdef class ExtensionMethod(ExtensionVirtualMethod):
    @staticmethod
    cdef void call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                        const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
                        GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) noexcept nogil:
        cdef Variant ret

        ret = ExtensionMethod._call(p_method_userdata, p_instance, p_args, p_argument_count, r_error)
        gdextension_interface_variant_new_copy(r_return, ret._native_ptr())


    @staticmethod
    cdef Variant _call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                            const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
                            GDExtensionCallError *r_error) noexcept with gil:
        cdef ExtensionMethod self = <object>p_method_userdata
        cdef Object instance = <object>p_instance
        cdef int i
        cdef list args = []
        cdef Variant arg
        for i in range(p_argument_count):
            arg = deref(<Variant *>p_args[i])
            args.append(arg.pythonize())
        # print("Extension method call %s%r" % (self.method, args))
        cdef object ret = self.method(instance, *args)
        if r_error:
            r_error[0].error = GDEXTENSION_CALL_OK
        return <Variant>ret


    @staticmethod
    cdef void ptrcall(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                      const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) noexcept nogil:
        ExtensionMethod._ptrcall(p_method_userdata, p_instance, p_args, r_return)


    @staticmethod
    cdef void _ptrcall(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                       const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) noexcept with gil:
        # TODO: Support any arg/return type, not just str/PythonObject
        cdef ExtensionMethod self = <object>p_method_userdata
        cdef Object wrapper = <object>p_instance
        cdef size_t i = 0
        cdef list args = []
        cdef String arg

        for i in range(self.get_argument_count() - 1):
            arg = deref(<String *>p_args[i])
            args.append(arg.py_str())

        cdef object ret = self.method(wrapper, *args)

        cdef PythonObject *gd_ret = PythonRuntime.get_singleton().python_object_from_pyobject(ret)
        (<void **>r_return)[0] = gd_ret._owner


    cdef int register(self) except -1:
        cdef GDExtensionClassMethodInfo mi

        if self.get_argument_count() < 1:
            raise TypeError('At least 1 argument ("self") is required')

        cdef PropertyInfo _return_value_info = self.get_return_info()
        cdef GDExtensionPropertyInfo return_value_info

        cdef PyStringName ret_name = PyStringName(_return_value_info.name)
        cdef PyStringName ret_classname = PyStringName(_return_value_info.class_name)
        cdef PyStringName ret_hintstring = PyStringName(_return_value_info.hint_string)

        return_value_info.type = <GDExtensionVariantType>(<VariantType>_return_value_info.type)
        return_value_info.name = ret_name.ptr()
        return_value_info.class_name = ret_classname.ptr()
        return_value_info.hint = _return_value_info.hint
        return_value_info.hint_string = ret_hintstring.ptr()
        return_value_info.usage = _return_value_info.usage

        # Skip self arg
        cdef list _arguments_info = self.get_argument_info_list()[1:]

        cdef size_t i = 0
        cdef size_t argsize = len(_arguments_info)

        cdef list _def_args = self.get_default_arguments()
        cdef GDExtensionVariantPtr *def_args = <GDExtensionVariantPtr *> \
            gdextension_interface_mem_alloc(len(_def_args) * cython.sizeof(GDExtensionVariantPtr))

        cdef Variant defarg
        for i in range(len(_def_args)):
            defarg = <Variant>_def_args[i]
            def_args[i] = <GDExtensionVariantPtr>&defarg
        
        cdef GDExtensionPropertyInfo *arguments_info = <GDExtensionPropertyInfo *> \
            gdextension_interface_mem_alloc(argsize * cython.sizeof(GDExtensionPropertyInfo))

        cdef list attr_names = [PyStringName(info.name) for info in _arguments_info]
        cdef list attr_classnames = [PyStringName(info.class_name) for info in _arguments_info]
        cdef list attr_hintstrings = [PyStringName(info.hint_string) for info in _arguments_info]

        for i in range(argsize):
            arguments_info[i].type = <GDExtensionVariantType>(<VariantType>_arguments_info[i].type)
            arguments_info[i].name = (<PyStringName>attr_names[i]).ptr()
            arguments_info[i].class_name = (<PyStringName>attr_classnames[i]).ptr()
            arguments_info[i].hint = _arguments_info[i].hint
            arguments_info[i].hint_string = (<PyStringName>attr_hintstrings[i]).ptr()
            arguments_info[i].usage = _arguments_info[i].usage

        cdef list _arguments_metadata = self.get_argument_metadata_list()[1:]
        cdef int *arguments_metadata = <int *>gdextension_interface_mem_alloc(len(_arguments_metadata) * cython.sizeof(int))
        for i in range(len(_arguments_metadata)):
            arguments_metadata[i] = <int>_arguments_metadata[i]

        cdef GDExtensionClassMethodArgumentMetadata return_value_metadata = \
            <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()

        cdef str method_name = self.__name__
        cdef StringName _method_name = StringName(method_name)

        # cdef tuple type_info = tuple([
        #     variant_type_to_str(<VariantType>return_value_info.type),
        #     *variant_type_to_str(<VariantType>arginfo.type) for arginfo in _arguments_info
        # ])

        # self.owner_class.__method_info__[method_name] = {'type_info': type_info, 'hash': None}

        mi.name = _method_name._native_ptr()
        mi.method_userdata = <void *><PyObject *>self
        mi.call_func = &ExtensionMethod.call
        mi.ptrcall_func = &ExtensionMethod.ptrcall
        mi.method_flags = self.get_hint_flags()
        mi.has_return_value = self.has_return()
        mi.return_value_info = &return_value_info
        mi.return_value_metadata = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE
        mi.argument_count = argsize
        mi.arguments_info = arguments_info
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>arguments_metadata
        mi.default_argument_count = self.get_default_argument_count()
        mi.default_arguments = def_args

        ref.Py_INCREF(self)
        self.owner_class._used_refs.append(self)

        cdef str class_name = self.owner_class.__name__
        cdef StringName _class_name = StringName(class_name)

        with nogil:
            gdextension_interface_classdb_register_extension_class_method(gdextension_library, _class_name._native_ptr(), &mi)

            gdextension_interface_mem_free(def_args)
            gdextension_interface_mem_free(arguments_info)
            gdextension_interface_mem_free(arguments_metadata)

        return 0
