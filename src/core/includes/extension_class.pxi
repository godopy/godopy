cdef dict _NODEDB = {}


cdef class ExtensionClass(Class):
    def __init__(self, name, object inherits, **kwargs):
        if not isinstance(inherits, (Class, str)):
            raise TypeError("'inherits' argument must be a Class instance or a string")

        self.__name__ = name
        if isinstance(inherits, Class):
            self.__inherits__ = inherits
        else:
            self.__inherits__ = Class.get_class(inherits)

        self.__method_info__ = {}

        self.is_registered = False
        self.is_virtual = kwargs.pop('is_virtual', False)
        self.is_abstract = kwargs.pop('is_abstract', False)
        self.is_exposed = kwargs.pop('is_exposed', True)
        self.is_runtime = kwargs.pop('is_runtime', False)

        self.method_bindings = {}
        self.python_method_bindings = {}
        self.virtual_method_bindings = {}
        self.virtual_method_implementation_bindings = {}

        self._used_refs = []


    def __call__(self):
        if not self.is_registered:
            raise RuntimeError("Extension class is not registered")
        return Extension(self, self.__inherits__)


    cdef tuple get_method_and_method_type_info(self, str name):
        cdef object method = self.virtual_method_implementation_bindings[name]
        cdef dict method_info = self.__inherits__.get_method_info(name)
        cdef tuple method_and_method_type_info = (method, method_info['type_info'])

        return method_and_method_type_info


    cdef void *get_method_and_method_type_info_ptr(self, str name) except NULL:
        cdef tuple method_and_method_type_info = self.get_method_and_method_type_info(name)

        self._used_refs.append(method_and_method_type_info)
        ref.Py_INCREF(method_and_method_type_info)

        return <void *><PyObject *>method_and_method_type_info


    cdef void *get_special_method_info_ptr(self, SpecialMethod method) except NULL:
        cdef tuple info = (method,)

        self._used_refs.append(info)
        ref.Py_INCREF(info)

        return <void *><PyObject *>info


    def bind_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required, got %s" % type(method))
        self.method_bindings[method.__name__] = method

        return method


    def bind_python_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required, got %s" % type(method))
        self.python_method_bindings[method.__name__] = method

        return method


    def bind_virtual_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required")
        self.virtual_method_implementation_bindings[method.__name__] = method

        return method


    def add_virtual_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required")
        self.virtual_method_bindings[method.__name__] = method

        return method


    def bind_methods(self, *methods):
        for method in methods:
            self.bind_method(method)


    def bind_virtual_methods(self, *methods):
        for method in methods:
            self.bind_virtual_method(method)


    cdef int set_registered(self) except -1:
        self.is_registered = True
        _CLASSDB[self.__name__] = self

        return 0


    def unregister(self) -> None:
        if not self.is_registered:
            if self.__name__ in _CLASSDB:
                del _CLASSDB[self.__name__]

            return

        # print("Unregistering Godot class %r" % self)

        del _CLASSDB[self.__name__]

        for reference in self._used_refs:
            ref.Py_DECREF(reference)

        self._used_refs = []

        ref.Py_DECREF(self)
        cdef PyStringName class_name = PyStringName(self.__name__)
        gdextension_interface_classdb_unregister_extension_class(gdextension_library, class_name.ptr())

        self.is_registered = False

        return 0


    def __del__(self):
        if self.is_registered:
            self.unregister()


    def register(self, **kwargs):
        try:
            return self._register(**kwargs)
        except Exception as exc:
            print_traceback_and_die(exc)

    def register_abstract(self):
        return self.register(is_abstract=True)



    def register_internal(self):
        return self.register(is_exposed=False)


    def register_runtime(self):
        return self.register(is_runtime=True)


    @staticmethod
    cdef GDExtensionObjectPtr create_instance(void *p_class_userdata,
                                              GDExtensionBool p_notify_postinitialize) noexcept nogil:
        return ExtensionClass._create_instance(p_class_userdata, p_notify_postinitialize)


    @staticmethod
    cdef GDExtensionObjectPtr _create_instance(void *p_self, bint p_notify_postinitialize) except? NULL with gil:
        if p_self == NULL:
            UtilityFunctions.push_error("ExtensionClass object pointer is uninitialized")
            return NULL

        from entry_point import get_config
        config = get_config()
        godot_class_to_class = config.get('godot_class_to_class')

        cdef ExtensionClass self = <ExtensionClass>p_self
        cls = godot_class_to_class(self)

        cdef Extension instance = cls(__godot_class__=self, from_callback=True, _internal_check=hex(<uint64_t>p_self))

        if self.__name__ in _NODEDB:
            UtilityFunctions.push_warning(
                "%s instance already saved to _NODEDB: %r, but another instance %r was requested, rewriting"
                % (self.__name__, _NODEDB[self.__name__], instance)
            )

            _NODEDB[self.__name__] = instance
        else:
            # print('Saved %r instance %r' % (self, instance))

            _NODEDB[self.__name__] = instance

        return instance._owner


    @staticmethod
    cdef void free_instance(void *data, void *p_instance) noexcept nogil:
        ExtensionClass._free_instance(data, p_instance)


    @staticmethod
    cdef int _free_instance(void *p_self, void *p_instance) except -1 with gil:
        cdef ExtensionClass self = <ExtensionClass>p_self
        cdef Extension instance = <Extension>p_instance

        # UtilityFunctions.print("Freeing %r" % instance)

        if self.__name__ in _NODEDB:
            del _NODEDB[self.__name__]

        ref.Py_DECREF(instance)

        return 0


    @staticmethod
    cdef GDExtensionObjectPtr recreate_instance(void *p_data, GDExtensionObjectPtr p_instance) noexcept nogil:
        with gil:
            print('ExtensinoClass recreate callback called; classs ptr:%x instance ptr:%x'
                  % (<uint64_t>p_data, <uint64_t>p_instance))
        return NULL


    def _register(self, **kwargs):
        if self.is_registered:
            raise RuntimeError("%r is already registered" % self)

        cdef GDExtensionClassCreationInfo4 ci
        cdef void *self_ptr = <PyObject *>self

        ci.is_virtual = kwargs.pop('is_virtual', self.is_virtual)
        ci.is_abstract = kwargs.pop('is_abstract', self.is_abstract)
        ci.is_exposed = kwargs.pop('is_exposed', self.is_exposed)
        ci.is_runtime = kwargs.pop('is_runtime', self.is_runtime)

        self.is_virtual = ci.is_virtual
        self.is_abstract = ci.is_abstract
        self.is_exposed = ci.is_exposed
        self.is_runtime = ci.is_runtime

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
        ci.class_userdata = self_ptr

        ref.Py_INCREF(self) # DECREF in ExtensionClass.unregister()

        # if kwargs.pop('has_get_property_list', False):
        #     ci.get_property_list_func = <GDExtensionClassGetPropertyList>&_ext_get_property_list_bind

        cdef StringName name = StringName(<const PyObject *>self.__name__)
        cdef StringName inherits_name = StringName(<const PyObject *>self.__inherits__.__name__)

        gdextension_interface_classdb_register_extension_class4(
            gdextension_library,
            name._native_ptr(),
            inherits_name._native_ptr(),
            &ci
        )

        for method in self.method_bindings.values():
            self.register_method(method)

        for method in self.virtual_method_bindings.values():
            self.register_virtual_method(method)

        self.set_registered()

        # print("%r is registered\n" % self)


    cdef int register_method(self, func: types.FunctionType) except -1:
        cdef ExtensionMethod method = ExtensionMethod(func)

        return method.register(self)


    cdef int register_virtual_method(self, func: types.FunctionType) except -1:
        cdef ExtensionVirtualMethod method = ExtensionVirtualMethod(func)

        return method.register(self)
