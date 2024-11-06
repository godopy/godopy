cdef class ExtensionMethod(_ExtensionMethodBase):
    """"
    Defines all custom methods of `gdextension.Extension` objects.

    Implements following GDExtension API calls:
        in `ExtensionMethod.register`
            `classdb_register_extension_class_method`

    Implements `call`/`ptrcall` callbacks in the `ClassMethodInfo` structure.
    """
    @staticmethod
    cdef void call_callback(void *p_method_userdata, void *p_instance, const (const void *) *p_args, int64_t p_count,
                            void *r_return, GDExtensionCallError *r_error) noexcept nogil:
        with gil:
            self = <object>p_method_userdata
            instance = <object>p_instance
            (<ExtensionMethod>self).call(
                instance,
                <const Variant **>p_args,
                p_count,
                <Variant *>r_return,
                r_error
            )

    cdef int call(self, object instance, const Variant **p_args, size_t p_count, Variant *r_ret,
                   GDExtensionCallError *r_error) except -1:

        cdef BoundPythonMethod method = BoundPythonMethod(instance, self.__func__)

        try:
            _make_python_varcall(method, p_args, p_count, r_ret, r_error)
        except Exception as exc:
            method.error_count += 1
            if method.error_count > 1:
                print_traceback_and_die(exc)
            else:
                print_error_with_traceback(exc)

    @staticmethod
    cdef void ptrcall_callback(void *p_method_userdata, void *p_instance, const (const void *) *p_args,
                               void *r_return) noexcept nogil:
        with gil:
            self = <object>p_method_userdata
            instance = <object>p_instance
            (<ExtensionMethod>self).ptrcall(instance, <const void **>p_args, <void *>r_return)

    cdef int ptrcall(self, object instance, const void **p_args, void *r_ret) except -1:
        cdef BoundPythonMethod method = BoundPythonMethod(instance, self)

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
        cdef _PropertyInfoData retinfo = _PropertyInfoData(py_retinfo)

        cdef list py_defargs = self.get_default_arguments()
        cdef _VariantPtrArray defargs = _VariantPtrArray(py_defargs)

        # Skip self arg
        cdef py_arginfo = self.get_argument_info_list()[1:]
        cdef _PropertyInfoDataArray arginfo = _PropertyInfoDataArray(py_arginfo)

        cdef list py_argmeta = self.get_argument_metadata_list()[1:]
        cdef _ArgumentMetadataArray argmeta = _ArgumentMetadataArray(py_argmeta)

        type_info = [variant_type_to_str(<VariantType>py_retinfo.type)]
        type_info += [variant_type_to_str(<VariantType>info.type) for info in py_arginfo]
        self.type_info = tuple(type_info)

        cdef StringName name = StringName(<const PyObject *>self.__name__)
        cdef StringName class_name = StringName(<const PyObject *>cls.__name__)

        # TODO: Support static, const, vararg and editor flags
        cdef GDExtensionClassMethodFlags flags = GDEXTENSION_METHOD_FLAG_NORMAL
        cdef void *self_ptr = <void *><PyObject *>self

        mi.name = name._native_ptr()
        mi.method_userdata = self_ptr
        mi.call_func = &ExtensionMethod.call_callback
        mi.ptrcall_func = &ExtensionMethod.ptrcall_callback
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
