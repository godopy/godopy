cdef class ExtensionMethod(_ExtensionMethodBase):
    @staticmethod
    cdef void call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                   const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
                   GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) noexcept nogil:
        ExtensionMethod._call(p_method_userdata, p_instance, <const Variant **>p_args, p_argument_count,
                              <Variant *>r_return, r_error)


    @staticmethod
    cdef void _call(void *p_method, void *p_self, const Variant **p_args, size_t p_count,
                    Variant *r_ret, GDExtensionCallError *r_error) noexcept with gil:
        cdef ExtensionMethod func = <object>p_method
        cdef Object instance = <object>p_self
        cdef BoundExtensionMethod method = BoundExtensionMethod(instance, func)

        # UtilityFunctions.print("Make python varcall %r %r" % (func, method))
        _make_python_varcall[BoundExtensionMethod](method, p_args, p_count, r_ret, r_error)


    @staticmethod
    cdef void ptrcall(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                      const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) noexcept nogil:
        ExtensionMethod._ptrcall(p_method_userdata, p_instance, <const void **>p_args, <void *>r_return)


    @staticmethod
    cdef void _ptrcall(void *p_method, GDExtensionClassInstancePtr p_self, const void **p_args,
                       void *r_ret) noexcept with gil:
        cdef ExtensionMethod func = <object>p_method
        cdef Object instance = <object>p_self
        cdef BoundExtensionMethod method = BoundExtensionMethod(instance, func)

        # UtilityFunctions.print("Make python ptrcall %r %r" % (func, method))
        _make_python_ptrcall[BoundExtensionMethod](method, r_ret, p_args, method.get_argument_count())


    cdef int register(self, ExtensionClass cls) except -1:
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

        type_info = [variant_type_to_str(<VariantType>return_value_info.type)]
        type_info += [variant_type_to_str(<VariantType>arginfo.type) for arginfo in _arguments_info]
        self.type_info = tuple(type_info)

        mi.name = _method_name._native_ptr()
        mi.method_userdata = <void *><PyObject *>self
        mi.call_func = &ExtensionMethod.call
        mi.ptrcall_func = &ExtensionMethod.ptrcall
        mi.method_flags = GDEXTENSION_METHOD_FLAG_NORMAL
        mi.has_return_value = self.has_return()
        mi.return_value_info = &return_value_info
        mi.return_value_metadata = <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()
        mi.argument_count = argsize
        mi.arguments_info = arguments_info
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>arguments_metadata
        mi.default_argument_count = self.get_default_argument_count()
        mi.default_arguments = def_args

        ref.Py_INCREF(self)
        cls._used_refs.append(self)

        cdef str class_name = cls.__name__
        cdef StringName _class_name = StringName(class_name)

        with nogil:
            gdextension_interface_classdb_register_extension_class_method(gdextension_library, _class_name._native_ptr(), &mi)

            gdextension_interface_mem_free(def_args)
            gdextension_interface_mem_free(arguments_info)
            gdextension_interface_mem_free(arguments_metadata)

            self.is_registered = True
