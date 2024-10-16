
cdef class ExtensionClassRegistrator:
    cdef str __name__
    cdef ExtensionClass registree
    cdef Class inherits
    cdef StringName _godot_class_name
    cdef StringName _godot_inherits_name

    def __cinit__(self, ExtensionClass registree, Class inherits, **kwargs):
        self.__name__ = registree.__name__
        self.registree = registree
        self.inherits = inherits

        if registree.is_registered:
            raise RuntimeError("%r is already registered" % registree)

        cdef GDExtensionClassCreationInfo4 *ci = \
            <GDExtensionClassCreationInfo4 *>gdextension_interface_mem_alloc(cython.sizeof(GDExtensionClassCreationInfo4))

        cdef void *registree_ptr = <PyObject *>registree

        ci.is_virtual = kwargs.pop('is_virtual', False)
        ci.is_abstract = kwargs.pop('is_abstract', False)
        ci.is_exposed = kwargs.pop('is_exposed', True)
        ci.is_runtime = kwargs.pop('is_runtime', False)
        ci.set_func = NULL # &_ext_set_bind
        ci.get_func = NULL # &_ext_.get_bind
        ci.get_property_list_func = NULL
        ci.free_property_list_func = NULL # &_ext_free_property_list_bind
        ci.property_can_revert_func = NULL # &_ext_property_can_revert_bind
        ci.property_get_revert_func = NULL # &_ext_property_get_revert_bind
        ci.validate_property_func = NULL # _ext_validate_property_bind
        ci.notification_func = NULL # &_ext_notification_bind
        ci.to_string_func = NULL # &_ext_to_string_bind
        ci.reference_func = NULL
        ci.unreference_func = NULL
        ci.create_instance_func = &ExtensionClass.create_instance
        ci.free_instance_func = &ExtensionClass.free_instance
        ci.recreate_instance_func = &ExtensionClass.recreate_instance
        ci.get_virtual_func = NULL
        ci.get_virtual_call_data_func = &Extension.get_virtual_call_data
        ci.call_virtual_with_data_func = &Extension.call_virtual_with_data
        ci.class_userdata = registree_ptr

        ref.Py_INCREF(self.registree) # DECREF in ExtensionClass.__dealoc__

        # if kwargs.pop('has_get_property_list', False):
        #     ci.get_property_list_func = <GDExtensionClassGetPropertyList>&_ext_get_property_list_bind

        cdef str name = self.__name__
        cdef str inherits_name = inherits.__name__
        cdef StringName _name = StringName(name)
        self._godot_class_name = StringName(_name)
        self._godot_inherits_name = StringName(inherits_name)

        with nogil:
            gdextension_interface_classdb_register_extension_class4(
                gdextension_library,
                self._godot_class_name._native_ptr(),
                self._godot_inherits_name._native_ptr(),
                ci
            )
            gdextension_interface_mem_free(ci)

        for method in registree.method_bindings.values():
            self.register_method(method)

        for method in registree.virtual_method_bindings.values():
            self.register_virtual_method(method)

        registree.set_registered()

        # print("%r is registered\n" % self.registree)


    cdef int register_method(self, func: types.FunctionType) except -1:
        cdef ExtensionMethod method = ExtensionMethod(self.registree, func)

        return method.register()


    cdef int register_virtual_method(self, func: types.FunctionType) except -1:
        cdef ExtensionVirtualMethod method = ExtensionVirtualMethod(self.registree, func)

        return method.register()
