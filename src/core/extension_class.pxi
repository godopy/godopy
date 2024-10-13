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


    cdef object get_method_and_method_type_info(self, str name):
        cdef object method = self.virtual_method_implementation_bindings[name]
        cdef dict method_info = self.__inherits__.get_method_info(name)
        cdef tuple method_and_method_type_info = (method, method_info['type_info'])

        return method_and_method_type_info


    cdef void *get_method_and_method_type_info_ptr(self, str name) except NULL:
        cdef tuple method_and_method_type_info = self.get_method_and_method_type_info(name)

        self._used_refs.append(method_and_method_type_info)
        ref.Py_INCREF(method_and_method_type_info)

        return <void *><PyObject *>method_and_method_type_info


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


    cdef void set_registered(self) noexcept nogil:
        self.is_registered = True


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
    cdef void _free_instance(void *p_self, void *p_instance) noexcept with gil:
        cdef ExtensionClass self = <ExtensionClass>p_self
        cdef Extension instance = <Extension>p_instance

        UtilityFunctions.print("Freeing %r and %r" % (self, instance))

        # for reference in self._used_refs:
        #     ref.Py_DECREF(reference)

        # self._used_refs = []

        ref.Py_DECREF(instance)
        # ref.Py_DECREF(self)


    @staticmethod
    cdef GDExtensionObjectPtr create_instance(void *p_class_userdata, GDExtensionBool p_notify_postinitialize) noexcept nogil:
        return ExtensionClass._create_instance(p_class_userdata, p_notify_postinitialize)


    @staticmethod
    cdef GDExtensionObjectPtr _create_instance(void *p_self, bint p_notify) except? NULL with gil:
        if p_self == NULL:
            UtilityFunctions.printerr("ExtensionClass object pointer is uninitialized")
            return NULL

        from godot import Extension as PyExtension

        cdef ExtensionClass self = <ExtensionClass>p_self
        cdef Extension instance = PyExtension(self, self.__inherits__, p_notify, True)

        return instance._owner


    @staticmethod
    cdef GDExtensionObjectPtr recreate_instance(void *p_data, GDExtensionObjectPtr p_instance) noexcept nogil:
        with gil:
            print('ExtensinoClass recreate callback called; classs ptr:%x instance ptr:%x' % (<uint64_t>p_data, <uint64_t>p_instance))
        return NULL
