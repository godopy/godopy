cdef inline void set_uint32_from_ptr(uint32_t *r_count, uint32_t value) noexcept nogil:
    cdef uint32_t count = cython.operator.dereference(r_count)
    count = value

cdef class GodotExtentionClass(godot.GodotClass):
    cdef godot.GodotClass parent

    @staticmethod
    cdef GDExtensionBool set_bind(GDExtensionClassInstancePtr p_instance,
                                  GDExtensionConstStringNamePtr p_name,
                                  GDExtensionConstVariantPtr p_value) noexcept nogil:
        if p_instance:
            # TODO: set instance property
            return False
        return False

    @staticmethod
    cdef GDExtensionBool get_bind(GDExtensionClassInstancePtr p_instance,
                                  GDExtensionConstStringNamePtr p_name,
                                  GDExtensionVariantPtr r_ret) noexcept nogil:
        if p_instance:
            # TODO: get instance property
            return False
        return False

    @staticmethod
    cdef bint has_get_property_list():
        # TODO: Check if a class has a property list
        return False

    @staticmethod
    cdef GDExtensionPropertyInfo *get_property_list_bind(GDExtensionClassInstancePtr p_instance,
                                                         uint32_t *r_count) noexcept nogil:
        if not p_instance:
            if r_count:
                set_uint32_from_ptr(r_count, 0)
            return NULL
        # TODO: Create and return property list
        if r_count:
            set_uint32_from_ptr(r_count, 0)
        return NULL

    @staticmethod
    cdef void free_property_list_bind(GDExtensionClassInstancePtr p_instance,
                                      const GDExtensionPropertyInfo *p_list,
                                      uint32_t p_count) noexcept nogil:
        if p_instance:
            pass

    @staticmethod
    cdef GDExtensionBool property_can_revert_bind(GDExtensionClassInstancePtr p_instance,
                                                  GDExtensionConstStringNamePtr p_name) noexcept nogil:
        return False

    @staticmethod
    cdef GDExtensionBool property_get_revert_bind(GDExtensionClassInstancePtr p_instance,
                                                  GDExtensionConstStringNamePtr p_name,
                                                  GDExtensionVariantPtr r_ret) noexcept nogil:
        return False

    @staticmethod
    cdef GDExtensionBool validate_property_bind(GDExtensionClassInstancePtr p_instance,
                                                GDExtensionPropertyInfo *p_property) noexcept nogil:
        return False

    @staticmethod
    cdef void notification_bind(GDExtensionClassInstancePtr p_instance,
                                int32_t p_what, GDExtensionBool p_reversed) noexcept nogil:
        pass

    @staticmethod
    cdef void to_string_bind(GDExtensionClassInstancePtr p_instance,
                             GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) noexcept nogil:
        pass

    @staticmethod
    cdef void free(void *data, GDExtensionClassInstancePtr ptr) noexcept nogil:
        pass

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance_func(void *data) noexcept nogil:
        with gil:
            return GodotExtentionClass._create_instance_func_gil(data)

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance_func_gil(void *data):
        print('_create_instance_func')
        cdef GodotExtentionClass cls = <object>data
        cdef GodotExtension wrapper = cls()
        return wrapper._owner

    @staticmethod
    cdef GDExtensionObjectPtr _recreate_instance_func(void *data, GDExtensionObjectPtr obj) noexcept nogil:
        return NULL

    @staticmethod
    cdef GDExtensionClassCallVirtual get_virtual_func(void *p_userdata,
                                                      GDExtensionConstStringNamePtr p_name) noexcept nogil:
        # TODO
        with gil:
            print('get_virtual_func')
        return NULL

    def __init__(self, name, godot.GodotClass parent, **kwargs):
        self.name = name
        self._name = stringname_from_str(name)
        self.parent = parent

        cdef GDExtensionClassCreationInfo3 ci

        ci.is_virtual = kwargs.pop('is_virtual', False)
        ci.is_abstract = kwargs.pop('is_abstract', False)
        ci.is_exposed = kwargs.pop('is_exposed', True)
        ci.is_runtime = kwargs.pop('is_runtime', False)
        ci.set_func = &GodotExtentionClass.set_bind
        ci.get_func = &GodotExtentionClass.get_bind
        ci.get_property_list_func = NULL
        ci.free_property_list_func = &GodotExtentionClass.free_property_list_bind
        ci.property_can_revert_func = &GodotExtentionClass.property_can_revert_bind
        ci.property_get_revert_func = &GodotExtentionClass.property_get_revert_bind
        ci.validate_property_func = &GodotExtentionClass.validate_property_bind
        ci.notification_func = &GodotExtentionClass.notification_bind
        ci.to_string_func = &GodotExtentionClass.to_string_bind
        ci.reference_func = NULL
        ci.unreference_func = NULL
        ci.create_instance_func = &GodotExtentionClass._create_instance_func
        ci.free_instance_func = &GodotExtentionClass.free
        ci.recreate_instance_func = &GodotExtentionClass._recreate_instance_func
        ci.get_virtual_func = &GodotExtentionClass.get_virtual_func
        ci.get_virtual_call_data_func = NULL
        ci.call_virtual_with_data_func = NULL
        ci.get_rid_func = NULL
        ci.class_userdata = <void *><PyObject *>self

        if GodotExtentionClass.has_get_property_list():
            ci.get_property_list_func = <GDExtensionClassGetPropertyList>&GodotExtentionClass.get_property_list_bind

        # defaults => register_class
        # is_abstract=True => register_abstract_class
        # is_exposed=False => register_internal_class
        # is_runtime=True => register_runtime_class

        _gde_classdb_register_extension_class3(gdextension_library,
                                               self._name._native_ptr(),
                                               parent._name._native_ptr(),
                                               &ci)

    def __call__(self):
        return GodotExtension(self.name)

    cpdef add_method(self, object func: types.FunctionType):
        cdef GodotExtensionMethod method = GodotExtensionMethod(self, func)
        cdef GDExtensionClassMethodInfo mi

        cdef GDExtensionVariantPtr *def_args = method.get_default_arguments()
        cdef GDExtensionPropertyInfo *return_value_info = method.get_argument_info_list()
        cdef GDExtensionPropertyInfo *arguments_info = return_value_info + 1
        cdef GDExtensionClassMethodArgumentMetadata *return_value_metadata = method.get_argument_metadata_list()
        cdef GDExtensionClassMethodArgumentMetadata *arguments_metadata = return_value_metadata + 1

        mi.name = method._name._native_ptr()
        mi.method_userdata = <void *><PyObject *>method
        mi.call_func = &GodotExtensionMethod.bind_call
        mi.ptrcall_func = &GodotExtensionMethod.bind_ptrcall
        mi.method_flags = method.get_hint_flags()
        mi.has_return_value = method.has_return()
        mi.return_value_info = return_value_info
        mi.return_value_metadata = cython.operator.dereference(return_value_metadata)
        mi.argument_count = method.get_argument_count()
        mi.arguments_info = arguments_info
        mi.arguments_metadata = arguments_metadata
        mi.default_argument_count = method.get_default_argument_count()
        mi.default_arguments = def_args

        ref.Py_INCREF(method)

        _gde_classdb_register_extension_class_method(gdextension_library, self._name._native_ptr(), &mi)
