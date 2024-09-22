cdef class PropertyInfo:
    cdef public VariantType type
    cdef public str name
    cdef public str class_name
    cdef public uint32_t hint
    cdef public str hint_string
    cdef public uint32_t usage

    def __cinit__(self, VariantType type, str name, str class_name, uint32_t hint=0, str hint_string='', uint32_t usage=0):
        self.type = type
        self.name = name
        self.class_name = class_name
        self.hint = hint
        self.hint_string = hint_string
        self.usage = usage

    def __repr__(self):
        return '<PropertyInfo %s:%s:%s:%d:%s:%d>' % (
            self.class_name, self.name, variant_to_str(self.type), self.hint, self.hint_string, self.usage)


cdef class ExtensionVirtualMethod:
    cdef ExtensionClass owner_class
    cdef object method
    cdef str __name__

    def __init__(self, ExtensionClass owner_class, object method: types.FunctionType):
        self.owner_class = owner_class
        self.method = method
        self.__name__ = method.__name__

    cdef list get_default_arguments(self):
        if self.method.__defaults__ is None:
            return []
        return [arg for arg in self.method.__defaults__]

    cdef PropertyInfo get_argument_info(self, int pos):
        cdef PropertyInfo pi = PropertyInfo(
            <int>GDEXTENSION_VARIANT_TYPE_FLOAT,
            '',
            self.owner_class.__name__
        )
        if pos >= 0:
            try:
                pi.name = self.method.__code__.co_varnames[pos]
            except IndexError:
                UtilityFunctions.push_error('Argname is missing in method %s, pos %d' % (self.method.__name__, pos))

        return pi

    cdef PropertyInfo get_return_info(self):
        return PropertyInfo(
            <int>GDEXTENSION_VARIANT_TYPE_NIL,
            '',
            self.owner_class.__name__
        )

    cdef list get_argument_info_list(self):
        return [self.get_argument_info(i) for i in range(self.get_argument_count())]

    cdef int get_return_metadata(self):
        return <int>GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE

    cdef list get_argument_metadata_list(self):
        cdef size_t i
        return [<int>GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE for i in range(self.get_argument_count())]

    cdef GDExtensionBool has_return(self):
        return <GDExtensionBool>bool(self.method.__annotations__.get('return'))

    cdef uint32_t get_hint_flags(self):
        return 0

    cdef uint32_t get_argument_count(self):
        return <uint32_t>self.method.__code__.co_argcount

    cdef uint32_t get_default_argument_count(self):
        if self.method.__defaults__ is None:
            return 0
        return <uint32_t>len(self.method.__defaults__)

    cdef int register(self) except -1:
        cdef GDExtensionClassVirtualMethodInfo mi

        if self.get_argument_count() < 1:
            raise RuntimeError('At least 1 argument ("self") is required')

        cdef PropertyInfo _return_value_info = self.get_return_info()
        cdef GDExtensionPropertyInfo return_value_info

        return_value_info.type = <GDExtensionVariantType>_return_value_info.type
        return_value_info.name = SN(_return_value_info.name).ptr()
        return_value_info.class_name = SN(_return_value_info.class_name).ptr()
        return_value_info.hint = _return_value_info.hint
        return_value_info.hint_string = SN(_return_value_info.hint_string).ptr() 
        return_value_info.usage = _return_value_info.usage

        # print('RETURN: %s' % _return_value_info)

        cdef size_t i

        cdef list _def_args = self.get_default_arguments()
        cdef GDExtensionVariantPtr *def_args = <GDExtensionVariantPtr *> \
            _gde_mem_alloc(len(_def_args) * cython.sizeof(GDExtensionVariantPtr))
        cdef Variant defarg
        for i in range(len(_def_args)):
            defarg = <Variant>_def_args[i]
            def_args[i] = <GDExtensionVariantPtr>&defarg

        # Skip self arg
        cdef list _arguments_info = self.get_argument_info_list()[1:]
        cdef size_t argsize = len(_arguments_info)
        cdef GDExtensionPropertyInfo *arguments_info = <GDExtensionPropertyInfo *> \
            _gde_mem_alloc(argsize * cython.sizeof(GDExtensionPropertyInfo))

        cdef str pyname
        cdef str pyclassname
        cdef str pyhintstring
        cdef int pytype

        for i in range(argsize):
            pyname = _arguments_info[i].name
            pytype = _arguments_info[i].type
            pyclassname = _arguments_info[i].class_name
            pyhintstring = _arguments_info[i].hint_string
            arguments_info[i].type = <GDExtensionVariantType>pytype
            arguments_info[i].name = (SN(pyname)).ptr()
            arguments_info[i].class_name = (SN(pyclassname)).ptr()
            arguments_info[i].hint = _arguments_info[i].hint
            arguments_info[i].hint_string = (SN(pyhintstring)).ptr()
            arguments_info[i].usage = _arguments_info[i].usage

        # print('ARGS: %s' % _arguments_info)

        cdef list _arguments_metadata = self.get_argument_metadata_list()[1:]
        cdef int *arguments_metadata = <int *>_gde_mem_alloc(len(_arguments_metadata) * cython.sizeof(int))
        for i in range(len(_arguments_metadata)):
            arguments_metadata[i] = <int>_arguments_metadata[i]

        cdef GDExtensionClassMethodArgumentMetadata return_value_metadata = \
            <GDExtensionClassMethodArgumentMetadata>self.get_return_metadata()

        cdef str method_name = self.__name__
        cdef StringName _method_name = StringName(method_name)

        mi.name = _method_name._native_ptr()
        mi.method_flags = self.get_hint_flags()
        mi.return_value = return_value_info
        mi.return_value_metadata = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE
        mi.argument_count = argsize
        mi.arguments = arguments_info
        mi.arguments_metadata = <GDExtensionClassMethodArgumentMetadata *>arguments_metadata

        print("REG VIRTUAL METHOD %s:%s %x" % (self.owner_class.__name__, self.__name__, <uint64_t><PyObject *>self))
        cdef str name = self.owner_class.__name__

        _gde_classdb_register_extension_class_virtual_method(gdextension_library, SN(name).ptr(), &mi)

        _gde_mem_free(def_args)
        _gde_mem_free(arguments_info)
        _gde_mem_free(arguments_metadata)
