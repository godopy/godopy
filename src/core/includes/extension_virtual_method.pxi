cdef class ExtensionVirtualMethod(_ExtensionMethodBase):
    cdef int register(self, ExtensionClass cls) except -1:
        cdef GDExtensionClassVirtualMethodInfo mi

        if self.get_argument_count() < 1:
            raise TypeError('At least 1 argument ("self") is required')

        cdef PropertyInfo py_retinfo = self.get_return_info()
        cdef _GDEPropInfoData return_value_info = _GDEPropInfoData(py_retinfo)

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
        mi.method_flags = GDEXTENSION_METHOD_FLAG_VIRTUAL
        mi.return_value = deref(return_value_info.ptr())
        mi.return_value_metadata = <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()
        mi.argument_count = arguments_info.propinfo_count
        mi.arguments = arguments_info.ptr()
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>argmeta_mem.ptr

        cdef PyStringName class_name = PyStringName(cls.__name__)

        gdextension_interface_classdb_register_extension_class_virtual_method(
            gdextension_library,
            class_name.ptr(),
            &mi
        )

        self.is_registered = True
