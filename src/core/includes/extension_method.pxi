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
        cdef _GDEPropInfoData return_value_info = _GDEPropInfoData(py_retinfo)

        cdef list _defargs = self.get_default_arguments()
        cdef size_t defarg_count = len(_defargs)
        cdef _Memory defarg_mem = _Memory(defarg_count * cython.sizeof(GDExtensionVariantPtr))

        cdef vector[Variant] defargs = vector[Variant]()
        defargs.resize(defarg_count)
        for i in range(defarg_count):
            type_funcs.variant_from_pyobject(_defargs[i], &defargs[i])
            (<Variant **>defarg_mem.ptr)[i] = &defargs[i]

        # Skip self arg
        cdef py_arginfo = self.get_argument_info_list()[1:]
        cdef _GDEPropInfoListData arguments_info = _GDEPropInfoListData(py_arginfo)

        cdef list arguments_metadata = self.get_argument_metadata_list()[1:]
        cdef size_t argmeta_count = len(arguments_metadata)
        cdef _Memory argmeta_mem = _Memory(argmeta_count * cython.sizeof(int))
        for i in range(argmeta_count):
            (<int *>argmeta_mem.ptr)[i] = <int>arguments_metadata[i]

        cdef PyStringName name = PyStringName(self.__name__)

        type_info = [variant_type_to_str(<VariantType>py_retinfo.type)]
        type_info += [variant_type_to_str(<VariantType>arginfo.type) for arginfo in py_arginfo]
        self.type_info = tuple(type_info)

        mi.name = name.ptr()
        mi.method_userdata = <void *><PyObject *>self
        mi.call_func = &ExtensionMethod.call
        mi.ptrcall_func = &ExtensionMethod.ptrcall
        mi.method_flags = GDEXTENSION_METHOD_FLAG_NORMAL
        mi.has_return_value = self.has_return()
        mi.return_value_info = return_value_info.ptr()
        mi.return_value_metadata = <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()
        mi.argument_count = arguments_info.propinfo_count
        mi.arguments_info = arguments_info.ptr()
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>argmeta_mem.ptr
        mi.default_argument_count = defarg_count
        mi.default_arguments = <GDExtensionVariantPtr *>defarg_mem.ptr

        ref.Py_INCREF(self)
        cls._used_refs.append(self)

        cdef PyStringName class_name = PyStringName(cls.__name__)
        gdextension_interface_classdb_register_extension_class_method(gdextension_library, class_name.ptr(), &mi)

        self.is_registered = True
