registry = {}

cdef class ExtensionClass(gd.Class):
    cdef gd.Class inherits
    cdef public bint is_registered
    cdef list method_registry

    @staticmethod
    cdef GDExtensionBool set_bind(GDExtensionClassInstancePtr p_instance,
                                  GDExtensionConstStringNamePtr p_name,
                                  GDExtensionConstVariantPtr p_value) noexcept nogil:
        if p_instance:
            # TODO: set instance property
            with gil:
                gd.__print('SET BIND INSTANCE %x' % <uint64_t>p_instance)
            return False
        return False

    @staticmethod
    cdef GDExtensionBool get_bind(GDExtensionClassInstancePtr p_instance,
                                  GDExtensionConstStringNamePtr p_name,
                                  GDExtensionVariantPtr r_ret) noexcept nogil:
        if p_instance:
            # TODO: get instance property
            with gil:
                gd.__print('GET BIND INSTANCE %x' % <uint64_t>p_instance)
            return False
        return False

    @staticmethod
    cdef bint has_get_property_list():
        # TODO: Check if a class has a property list
        return False

    @staticmethod
    cdef GDExtensionPropertyInfo *get_property_list_bind(GDExtensionClassInstancePtr p_instance,
                                                         uint32_t *r_count) noexcept nogil:
        with gil:
            print('GETPROPLIST %x' % (<uint64_t>p_instance))
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
            with gil:
                print('FREEPROPLIST %x' % (<uint64_t>p_instance))

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
        if p_instance:
            with gil:
                print('NOTOFICATION BIND %x' % (<uint64_t>p_instance))

    @staticmethod
    cdef void to_string_bind(GDExtensionClassInstancePtr p_instance,
                             GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) noexcept nogil:
        if p_instance:
            with gil:
                print('TO STRING BIND %x' % (<uint64_t>p_instance))

    @staticmethod
    cdef void free(void *data, GDExtensionClassInstancePtr p_instance) noexcept nogil:
        with gil:
            print('FREE %x %x' % (<uint64_t>data, <uint64_t>p_instance))
            ExtensionClass._free(data, p_instance)

    @staticmethod
    cdef int _free(void *p_class, void *p_instance) except -1:
        cdef Extension wrapper = <Extension>p_instance
        ref.Py_DECREF(wrapper)

        cdef ExtensionClass cls = <ExtensionClass>p_class
        ref.Py_DECREF(cls)

        return 0

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance_func(void *data) noexcept nogil:
        with gil:
            return ExtensionClass._create_instance_func_gil(data)

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance_func_gil(void *data):
        print('CREATE INSTANCE %x' % <uint64_t>data)

        cdef ExtensionClass cls = <ExtensionClass>data

        # cdef ExtensionClass creating = registry[cls.__name__]
        cdef gd.Class base = cls.inherits

        cdef Extension wrapper = Extension(base, cls)

        print("CREATED INSTANCE %r %x %x" % (wrapper, <uint64_t>wrapper._owner, <uint64_t><PyObject *>wrapper))

        return wrapper._owner

    @staticmethod
    cdef GDExtensionObjectPtr _recreate_instance_func(void *data, GDExtensionObjectPtr obj) noexcept nogil:
        with gil:
            print('RECREATE %x %x' % (<uint64_t>data, <uint64_t>obj))
        return NULL

    @staticmethod
    cdef GDExtensionClassCallVirtual get_virtual_func(void *p_userdata,
                                                      GDExtensionConstStringNamePtr p_name) noexcept nogil:

        cdef StringName name = deref(<StringName *>p_name) 
        with gil:
            print('GETVIRTUAL %x %s' % (<uint64_t>p_userdata, name.py_str()))
            ExtensionClass._get_virtual_func(p_userdata, name.py_str())

    @staticmethod
    cdef GDExtensionClassCallVirtual _get_virtual_func(void *p_cls, str name):
        # cdef ExtensionClass cls = <ExtensionClass>p_cls
        if name == '_process':
            print("RETURN VURTUALFUNC")
            return ExtensionClass.virtualfunc


    @staticmethod
    cdef void virtualfunc(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) noexcept nogil:
       with gil:
            ExtensionClass._virtualfunc(p_instance, p_args, r_ret)

    @staticmethod
    cdef void _virtualfunc(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret):
        cdef object wrapper = <object>p_instance
        print('virtualmethod call %r' % wrapper)

    def __init__(self, name, object inherits, **kwargs):
        if not isinstance(inherits, (gd.Class, str)):
            raise TypeError("'inherits' argument must be a Class instance or a string")

        self.__name__ = name
        if isinstance(inherits, gd.Class):
            self.inherits = inherits
        else:
            self.inherits = gd.Class(inherits)

        self.is_registered = False

        self.method_registry = []

    def __call__(self):
        if not self.is_registered:
            raise RuntimeError("Extension class is not registered")
        return Extension(self.__name__)

    def add_method(self, method: types.FunctionType):
        if not isinstance(method, types.FunctionType):
            raise TypeError("Function is required")
        self.method_registry.append(method)

    def add_methods(self, *methods):
        for method in methods:
            self.add_method(method)

    cdef set_registered(self):
        self.is_registered = True

    cpdef register(self):
        return ExtensionClassRegistrator(self, self.inherits)

    cpdef register_abstract(self):
        return ExtensionClassRegistrator(self, self.inherits, is_abstract=True)
    
    cpdef register_internal(self):
        return ExtensionClassRegistrator(self, self.inherits, is_exposed=False)

    cpdef register_runtime(self):
        return ExtensionClassRegistrator(self, self.inherits, is_runtime=True)


cdef inline void set_uint32_from_ptr(uint32_t *r_count, uint32_t value) noexcept nogil:
    cdef uint32_t count = cython.operator.dereference(r_count)
    count = value

cdef class ExtensionClassRegistrator:
    cdef str __name__
    cdef ExtensionClass registree
    cdef gd.Class inherits
    cdef StringName _godot_class_name
    cdef StringName _godot_inherits_name

    def __cinit__(self, ExtensionClass registree, gd.Class inherits, **kwargs):
        self.__name__ = registree.__name__
        self.registree = registree
        self.inherits = inherits

        if registree.is_registered:
            raise RuntimeError("%r is already registered" % registree)

        cdef GDExtensionClassCreationInfo3 ci

        ci.is_virtual = kwargs.pop('is_virtual', False)
        ci.is_abstract = kwargs.pop('is_abstract', False)
        ci.is_exposed = kwargs.pop('is_exposed', True)
        ci.is_runtime = kwargs.pop('is_runtime', False)
        ci.set_func = NULL # &ExtensionClass.set_bind
        ci.get_func = NULL # &ExtensionClass.get_bind
        ci.get_property_list_func = NULL
        ci.free_property_list_func = &ExtensionClass.free_property_list_bind
        ci.property_can_revert_func = &ExtensionClass.property_can_revert_bind
        ci.property_get_revert_func = &ExtensionClass.property_get_revert_bind
        ci.validate_property_func = ExtensionClass.validate_property_bind
        ci.notification_func = NULL # &ExtensionClass.notification_bind
        ci.to_string_func = &ExtensionClass.to_string_bind
        ci.reference_func = NULL
        ci.unreference_func = NULL
        ci.create_instance_func = &ExtensionClass._create_instance_func
        ci.free_instance_func = &ExtensionClass.free
        ci.recreate_instance_func = &ExtensionClass._recreate_instance_func
        ci.get_virtual_func = ExtensionClass.get_virtual_func
        ci.get_virtual_call_data_func = NULL
        ci.call_virtual_with_data_func = NULL
        ci.get_rid_func = NULL
        ci.class_userdata = <void *><PyObject *>self.registree

        ref.Py_INCREF(self.registree) # DECREF in ExtenstionClass._free

        # print('Set USERDATA %x' % <uint64_t>ci.class_userdata)

        if ExtensionClass.has_get_property_list():
            ci.get_property_list_func = <GDExtensionClassGetPropertyList>&ExtensionClass.get_property_list_bind

        # defaults => register_class
        # is_abstract=True => register_abstract_class
        # is_exposed=False => register_internal_class
        # is_runtime=True => register_runtime_class

        cdef str name = self.__name__
        cdef str inherits_name = inherits.__name__
        cdef StringName extra = StringName(name)
        self._godot_class_name = StringName(extra)
        self._godot_inherits_name = StringName(inherits_name)
        _gde_classdb_register_extension_class3(
            gdextension_library,
            &self._godot_class_name,
            &self._godot_inherits_name,
            &ci
        )

        for method in registree.method_registry:
            self.register_method(method)

        registree.set_registered()
        registry[self.__name__] = registree
        print("%r registered" % self.__name__)

    def register_method(self, func: types.FunctionType):
        print("registering method %r" % func)
        cdef ExtensionMethod method = ExtensionMethod(self.registree, func)
        cdef GDExtensionClassMethodInfo mi

        if method.get_argument_count() < 1:
            raise RuntimeError('At least 1 argument ("self") is required')

        cdef PropertyInfo _return_value_info = method.get_return_info()
        cdef GDExtensionPropertyInfo return_value_info

        return_value_info.type = <GDExtensionVariantType>_return_value_info.type
        return_value_info.name = SN(_return_value_info.name).ptr()
        return_value_info.class_name = SN(_return_value_info.class_name).ptr()
        return_value_info.hint = _return_value_info.hint
        return_value_info.hint_string = SN(_return_value_info.hint_string).ptr() 
        return_value_info.usage = _return_value_info.usage

        cdef size_t i

        cdef list _def_args = method.get_default_arguments()
        cdef GDExtensionVariantPtr *def_args = <GDExtensionVariantPtr *> \
            _gde_mem_alloc(len(_def_args) * cython.sizeof(GDExtensionVariantPtr))
        cdef Variant defarg
        for i in range(len(_def_args)):
            defarg = <Variant>_def_args[i]
            def_args[i] = <GDExtensionVariantPtr>&defarg

        # Skip self arg
        cdef list _arguments_info = method.get_argument_info_list()[1:]
        cdef size_t argsize = len(_arguments_info)
        cdef GDExtensionPropertyInfo *arguments_info = <GDExtensionPropertyInfo *> \
            _gde_mem_alloc(argsize * cython.sizeof(GDExtensionPropertyInfo))

        cdef str pyname
        cdef str pyclassname
        cdef str pyhintstring
        cdef int pytype

        for i in range(argsize):
            pyname = _arguments_info[i].name
            pyclassname = _arguments_info[i].class_name
            pyhintstring = _arguments_info[i].hint_string
            pytype = _arguments_info[i].type
            arguments_info[i].type = <GDExtensionVariantType>pytype
            arguments_info[i].name = (SN(pyname)).ptr()
            arguments_info[i].class_name = (SN(pyclassname)).ptr()
            arguments_info[i].hint = _arguments_info[i].hint
            arguments_info[i].hint_string = (SN(pyhintstring)).ptr() 
            arguments_info[i].usage = _arguments_info[i].usage

        cdef list _arguments_metadata = method.get_argument_metadata_list()[1:]
        cdef int *arguments_metadata = <int *>_gde_mem_alloc(len(_arguments_metadata) * cython.sizeof(int))
        for i in range(len(_arguments_metadata)):
            arguments_metadata[i] = <int>_arguments_metadata[i]

        cdef GDExtensionClassMethodArgumentMetadata return_value_metadata = \
            <GDExtensionClassMethodArgumentMetadata>method.get_return_metadata()

        cdef str method_name = method.__name__
        cdef StringName _method_name = StringName(method_name)

        mi.name = _method_name._native_ptr()
        mi.method_userdata = <void *><PyObject *>method
        mi.call_func = &ExtensionMethod.bind_call
        mi.ptrcall_func = &ExtensionMethod.bind_ptrcall
        mi.method_flags = method.get_hint_flags()
        mi.has_return_value = method.has_return()
        mi.return_value_info = NULL # &return_value_info
        mi.return_value_metadata = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE
        mi.argument_count = argsize
        mi.arguments_info = arguments_info
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>arguments_metadata
        mi.default_argument_count = method.get_default_argument_count()
        mi.default_arguments = def_args

        ref.Py_INCREF(method)

        print("REG METHOD %s:%s %x" % (self.__name__, method.__name__, <uint64_t><PyObject *>method))
        cdef str name = self.__name__

        _gde_classdb_register_extension_class_method(gdextension_library, SN(name).ptr(), &mi)

        _gde_mem_free(def_args)
        _gde_mem_free(arguments_info)
        _gde_mem_free(arguments_metadata)
