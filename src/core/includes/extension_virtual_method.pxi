cdef class ExtensionVirtualMethod(_ExtensionMethodBase):
    cdef int register(self, ExtensionClass cls) except -1:
        cdef GDExtensionClassVirtualMethodInfo mi

        if self.get_argument_count() < 1:
            raise TypeError('At least 1 argument ("self") is required')

        cdef PropertyInfo py_retinfo = self.get_return_info()
        cdef _PropertyInfoData retinfo = _PropertyInfoData(py_retinfo)

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
        cdef GDExtensionClassMethodFlags flags = GDEXTENSION_METHOD_FLAG_VIRTUAL

        mi.name = name._native_ptr()
        mi.method_flags = flags
        mi.return_value = deref(retinfo.ptr())
        mi.return_value_metadata = <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()
        mi.argument_count = arginfo.count
        mi.arguments = arginfo.ptr()
        mi.arguments_metadata = argmeta.ptr()

        gdextension_interface_classdb_register_extension_class_virtual_method(
            gdextension_library,
            class_name._native_ptr(),
            &mi
        )

        self.is_registered = True
