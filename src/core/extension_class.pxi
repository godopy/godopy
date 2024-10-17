cdef dict _NODEDB = {}
cdef list _registered_classes = []


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
        _registered_classes.append(self)

        return 0


    cdef int unregister(self) except -1:
        if not self.is_registered:
            return 0

        # print("Unregistering Godot class %r" % self)

        for reference in self._used_refs:
            ref.Py_DECREF(reference)

        self._used_refs = []

        ref.Py_DECREF(self)
        cdef StringName class_name = StringName(self.__name__)
        gdextension_interface_classdb_unregister_extension_class(gdextension_library, class_name._native_ptr())

        self.is_registered = False

        return 0


    def __del__(self):
        if self.is_registered:
            self.unregister()


    def register(self):
        return ExtensionClassRegistrator(self, self.__inherits__)


    def register_abstract(self):
        return ExtensionClassRegistrator(self, self.__inherits__, is_abstract=True)


    def register_internal(self):
        return ExtensionClassRegistrator(self, self.__inherits__, is_exposed=False)


    def register_runtime(self):
        return ExtensionClassRegistrator(self, self.__inherits__, is_runtime=True)


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
    cdef GDExtensionObjectPtr create_instance(void *p_class_userdata,
                                              GDExtensionBool p_notify_postinitialize) noexcept nogil:
        return ExtensionClass._create_instance(p_class_userdata, p_notify_postinitialize)


    @staticmethod
    cdef GDExtensionObjectPtr _create_instance(void *p_self, bint p_notify_postinitialize) except? NULL with gil:
        if p_self == NULL:
            UtilityFunctions.printerr("ExtensionClass object pointer is uninitialized")
            return NULL

        from godot import Extension as PyExtension

        cdef ExtensionClass self = <ExtensionClass>p_self
        cdef Extension instance = PyExtension(self, self.__inherits__, p_notify_postinitialize, True)

        assert self.__name__ not in _NODEDB
        # print('Saved %r instance %r' % (self, instance))
        _NODEDB[self.__name__] = instance

        return instance._owner


    @staticmethod
    cdef GDExtensionObjectPtr recreate_instance(void *p_data, GDExtensionObjectPtr p_instance) noexcept nogil:
        with gil:
            print('ExtensinoClass recreate callback called; classs ptr:%x instance ptr:%x'
                  % (<uint64_t>p_data, <uint64_t>p_instance))
        return NULL
