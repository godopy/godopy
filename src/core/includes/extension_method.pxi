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

        try:
            _make_python_varcall(method, p_args, p_count, r_ret, r_error)
        except Exception as exc:
            method.error_count += 1
            if method.error_count > 1:
                print_traceback_and_die(exc)
            else:
                print_error_with_traceback(exc)

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

        try:
            _make_python_ptrcall(method, r_ret, p_args, method.get_argument_count())
        except Exception as exc:
            method.error_count += 1
            if method.error_count > 1:
                print_traceback_and_die(exc)
            else:
                print_error_with_traceback(exc)


    cdef int register(self, ExtensionClass cls) except -1:
        cdef GDExtensionClassMethodInfo mi

        if self.get_argument_count() < 1:
            raise TypeError('At least 1 argument ("self") is required')

        cdef PropertyInfo py_retinfo = self.get_return_info()
        cdef _GDEPropInfoData retinfo = _GDEPropInfoData(py_retinfo)

        cdef list py_defargs = self.get_default_arguments()
        cdef _VariantPtrArray defargs = _VariantPtrArray(py_defargs)

        # Skip self arg
        cdef py_arginfo = self.get_argument_info_list()[1:]
        cdef _GDEPropInfoListData arginfo = _GDEPropInfoListData(py_arginfo)

        cdef list py_argmeta = self.get_argument_metadata_list()[1:]
        cdef _GDEArgumentMetadataArray argmeta = _GDEArgumentMetadataArray(py_argmeta)

        type_info = [variant_type_to_str(<VariantType>py_retinfo.type)]
        type_info += [variant_type_to_str(<VariantType>info.type) for info in py_arginfo]
        self.type_info = tuple(type_info)

        cdef StringName name = StringName(<const PyObject *>self.__name__)
        cdef StringName class_name = StringName(<const PyObject *>cls.__name__)

        # TODO: Support static, const, vararg and editor flags
        cdef GDExtensionClassMethodFlags flags = GDEXTENSION_METHOD_FLAG_NORMAL

        mi.name = name._native_ptr()
        mi.method_userdata = <void *><PyObject *>self
        mi.call_func = &ExtensionMethod.call
        mi.ptrcall_func = &ExtensionMethod.ptrcall
        mi.method_flags = flags
        mi.has_return_value = self.has_return()
        mi.return_value_info = retinfo.ptr()
        mi.return_value_metadata = <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()
        mi.argument_count = arginfo.count
        mi.arguments_info = arginfo.ptr()
        mi.arguments_metadata = argmeta.ptr()
        mi.default_argument_count = defargs.count
        mi.default_arguments = <GDExtensionVariantPtr *>defargs.ptr()

        ref.Py_INCREF(self)
        cls._used_refs.append(self)

        gdextension_interface_classdb_register_extension_class_method(
            gdextension_library,
            class_name._native_ptr(),
            &mi
        )

        self.is_registered = True
