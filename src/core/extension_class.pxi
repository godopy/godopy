registry = {}


cdef class ExtensionClass(Class):
    cdef readonly bint is_registered
    cdef readonly dict method_bindings
    cdef readonly dict virtual_method_bindings
    cdef readonly dict virtual_method_implementation_bindings


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
        self.virtual_method_bindings = {}
        self.virtual_method_implementation_bindings = {}


    def __call__(self):
        if not self.is_registered:
            raise RuntimeError("Extension class is not registered")
        return Extension(self.__name__)


    def bind_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required")
        self.method_bindings[method.__name__] = method

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


    cdef set_registered(self):
        self.is_registered = True


    cpdef register(self):
        return ExtensionClassRegistrator(self, self.__inherits__)


    cpdef register_abstract(self):
        return ExtensionClassRegistrator(self, self.__inherits__, is_abstract=True)


    cpdef register_internal(self):
        return ExtensionClassRegistrator(self, self.__inherits__, is_exposed=False)


    cpdef register_runtime(self):
        return ExtensionClassRegistrator(self, self.__inherits__, is_runtime=True)


    @staticmethod
    cdef void free_instance(void *data, void *p_instance) noexcept nogil:
        with gil:
            print('EXT CLASS FREE %x %x' % (<uint64_t>data, <uint64_t>p_instance))
            ExtensionClass._free_instance(data, p_instance)


    @staticmethod
    cdef int _free_instance(void *p_class, void *p_instance) except -1:
        cdef Extension wrapper = <Extension>p_instance
        ref.Py_DECREF(wrapper)

        cdef ExtensionClass cls = <ExtensionClass>p_class
        ref.Py_DECREF(cls)

        return 0


    @staticmethod
    cdef GDExtensionObjectPtr create_instance(
        void *p_class_userdata,
        GDExtensionBool p_notify_postinitialize
    ) noexcept nogil:
        with gil:
            return ExtensionClass._create_instance(p_class_userdata, p_notify_postinitialize)


    @staticmethod
    cdef GDExtensionObjectPtr _create_instance(void *data, bint notify) except NULL:
        print('CREATE INSTANCE %x %s' % (<uint64_t>data, notify))

        if data == NULL:
            UtilityFunctions.printerr("ExtensionClass object pointer is uninitialized")
            return NULL

        cdef ExtensionClass cls = <ExtensionClass>data
        cdef Class base = cls.__inherits__
        cdef Extension wrapper = Extension(base, cls, notify)

        print("CREATED INSTANCE %r %x %x" % (wrapper, <uint64_t>wrapper._owner, <uint64_t><PyObject *>wrapper))

        return wrapper._owner


    @staticmethod
    cdef GDExtensionObjectPtr recreate_instance(void *data, GDExtensionObjectPtr obj) noexcept nogil:
        with gil:
            print('RECREATE %x %x' % (<uint64_t>data, <uint64_t>obj))
        return NULL
