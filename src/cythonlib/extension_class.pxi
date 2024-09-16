cdef class ExtensionClass(gd.Class):
    cdef gd.Class parent
    cdef public bint is_registered
    cdef vector[String] method_registry

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
            return ExtensionClass._create_instance_func_gil(data)

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance_func_gil(void *data):
        print('_create_instance_func')
        cdef ExtensionClass cls = <object>data
        cdef Extension wrapper = cls()
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

    def __init__(self, name, object parent, **kwargs):
        if not isinstance(parent, (gd.Class, str)):
            raise TypeError("'parent' argument must be a Class instance or a string")

        self.__name__ = name
        if isinstance(parent, gd.Class):
            self.parent = parent
        else:
            self.parent = gd.Class(str(parent))

        self.is_registered = False
        # self.method_registry = []

    def __call__(self):
        if not self.is_registered:
            raise RuntimeError("Extension class not registered")
        return Extension(self.__name__)

    def add_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required")
        self.method_registry.push_back(<String>method)

    def add_methods(self, *methods):
        for method in methods:
            self.add_method(method)

    cdef set_registered(self):
        self.is_registered = True


cdef inline void set_uint32_from_ptr(uint32_t *r_count, uint32_t value) noexcept nogil:
    cdef uint32_t count = cython.operator.dereference(r_count)
    count = value


cdef class ExtensionClassRegistrator:
    cdef gd.Class parent
    cdef ExtensionClass registree

    def __cinit__(self, ExtensionClass registree, gd.Class parent, **kwargs):
        self.name = registree.__name__
        self.registree = registree
        self.parent = parent

        if registree.is_registered:
            raise RuntimeError("%r is already registered" % registree)

        cdef GDExtensionClassCreationInfo3 ci

        ci.is_virtual = kwargs.pop('is_virtual', False)
        ci.is_abstract = kwargs.pop('is_abstract', False)
        ci.is_exposed = kwargs.pop('is_exposed', True)
        ci.is_runtime = kwargs.pop('is_runtime', False)
        ci.set_func = &ExtensionClass.set_bind
        ci.get_func = &ExtensionClass.get_bind
        ci.get_property_list_func = NULL
        ci.free_property_list_func = &ExtensionClass.free_property_list_bind
        ci.property_can_revert_func = &ExtensionClass.property_can_revert_bind
        ci.property_get_revert_func = &ExtensionClass.property_get_revert_bind
        ci.validate_property_func = &ExtensionClass.validate_property_bind
        ci.notification_func = &ExtensionClass.notification_bind
        ci.to_string_func = &ExtensionClass.to_string_bind
        ci.reference_func = NULL
        ci.unreference_func = NULL
        ci.create_instance_func = &ExtensionClass._create_instance_func
        ci.free_instance_func = &ExtensionClass.free
        ci.recreate_instance_func = &ExtensionClass._recreate_instance_func
        ci.get_virtual_func = &ExtensionClass.get_virtual_func
        ci.get_virtual_call_data_func = NULL
        ci.call_virtual_with_data_func = NULL
        ci.get_rid_func = NULL
        ci.class_userdata = <void *><PyObject *>self

        if ExtensionClass.has_get_property_list():
            ci.get_property_list_func = <GDExtensionClassGetPropertyList>&ExtensionClass.get_property_list_bind

        # defaults => register_class
        # is_abstract=True => register_abstract_class
        # is_exposed=False => register_internal_class
        # is_runtime=True => register_runtime_class

        cdef str name = self.__name__
        cdef str parent_name = parent.__name__
        _gde_classdb_register_extension_class3(gdextension_library,
                                               StringName(name)._native_ptr(),
                                               StringName(parent_name)._native_ptr(),
                                               &ci)

        cdef size_t i = 0
        for i in range(registree.method_registry.size()):
            method = registree.method_registry[i]
            self.register_method(method.py_str())

        registree.set_registered()

    cdef register_method(self, object func: types.FunctionType):
        cdef ExtensionMethod method = ExtensionMethod(self, func)
        cdef GDExtensionClassMethodInfo mi

        cdef GDExtensionVariantPtr *def_args = method.get_default_arguments()
        cdef GDExtensionPropertyInfo *return_value_info = method.get_argument_info_list()
        cdef GDExtensionPropertyInfo *arguments_info = return_value_info + 1
        cdef GDExtensionClassMethodArgumentMetadata *return_value_metadata = method.get_argument_metadata_list()
        cdef GDExtensionClassMethodArgumentMetadata *arguments_metadata = return_value_metadata + 1

        mi.name = StringName(method.__name__)._native_ptr()
        mi.method_userdata = <void *><PyObject *>method
        mi.call_func = &ExtensionMethod.bind_call
        mi.ptrcall_func = &ExtensionMethod.bind_ptrcall
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

        cdef str name = self.__name__
        _gde_classdb_register_extension_class_method(
            gdextension_library, StringName(name)._native_ptr(), &mi)
