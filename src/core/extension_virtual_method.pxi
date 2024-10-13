cdef class PropertyInfo:
    cdef public VariantType type
    cdef public str name
    cdef public str class_name
    cdef public uint32_t hint
    cdef public str hint_string
    cdef public uint32_t usage

    def __cinit__(self, VariantType type, str name='', str class_name='', uint32_t hint=0, str hint_string='', uint32_t usage=0):
        self.type = type
        self.name = name
        self.class_name = class_name
        self.hint = hint
        self.hint_string = hint_string
        self.usage = usage

    def __repr__(self):
        cls_name = '%s.' % self.class_name if self.class_name else ''
        return '<PropertyInfo %s%s:%s>' % (cls_name, self.name, variant_type_to_str(self.type))


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
        cdef PropertyInfo pi = PropertyInfo(NIL)

        if pos >= 0:
            try:
                pi.name = self.method.__code__.co_varnames[pos]
                pi.type = pytype_to_gdtype(self.method.__annotations__.get(pi.name, None))
            except IndexError:
                UtilityFunctions.push_error('Argname is missing in method %s, pos %d' % (self.method.__name__, pos))

        return pi

    cdef PropertyInfo get_return_info(self):
        return PropertyInfo(
            pytype_to_gdtype(self.method.__annotations__.get('return', None)),
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
            raise TypeError('At least 1 argument ("self") is required')

        cdef PropertyInfo _return_value_info = self.get_return_info()
        cdef GDExtensionPropertyInfo return_value_info

        cdef str pyname = _return_value_info.name
        cdef str pyclassname = _return_value_info.class_name
        cdef str pyhintstring = _return_value_info.hint_string
        cdef int pytype = _return_value_info.type

        cdef StringName _name = StringName(pyname)
        cdef StringName classname = StringName(pyclassname)
        cdef StringName hintstring = StringName(pyhintstring)

        return_value_info.type = <GDExtensionVariantType>pytype
        return_value_info.name = _name._native_ptr()
        return_value_info.class_name = classname._native_ptr()
        return_value_info.hint = _return_value_info.hint
        return_value_info.hint_string = hintstring._native_ptr()
        return_value_info.usage = _return_value_info.usage

        cdef size_t i

        cdef list _def_args = self.get_default_arguments()
        cdef GDExtensionVariantPtr *def_args = <GDExtensionVariantPtr *> \
            gdextension_interface_mem_alloc(len(_def_args) * cython.sizeof(GDExtensionVariantPtr))
        cdef Variant defarg
        for i in range(len(_def_args)):
            defarg = <Variant>_def_args[i]
            def_args[i] = <GDExtensionVariantPtr>&defarg

        # Skip self arg
        cdef list _arguments_info = self.get_argument_info_list()[1:]
        cdef size_t argsize = len(_arguments_info)
        cdef GDExtensionPropertyInfo *arguments_info = <GDExtensionPropertyInfo *> \
            gdextension_interface_mem_alloc(argsize * cython.sizeof(GDExtensionPropertyInfo))


        for i in range(argsize):
            pyname = _arguments_info[i].name
            pyclassname = _arguments_info[i].class_name
            pyhintstring = _arguments_info[i].hint_string
            pytype = _arguments_info[i].type
            _name = StringName(pyname)
            classname = StringName(pyclassname)
            hintstring = StringName(pyhintstring)
            arguments_info[i].type = <GDExtensionVariantType>pytype
            arguments_info[i].name = _name._native_ptr()
            arguments_info[i].class_name = classname._native_ptr()
            arguments_info[i].hint = _arguments_info[i].hint
            arguments_info[i].hint_string = hintstring._native_ptr()
            arguments_info[i].usage = _arguments_info[i].usage

        cdef list _arguments_metadata = self.get_argument_metadata_list()[1:]
        cdef int *arguments_metadata = \
            <int *>gdextension_interface_mem_alloc(len(_arguments_metadata) * cython.sizeof(int))
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

        cdef str name = self.owner_class.__name__
        cdef StringName _class_name = StringName(name)

        gdextension_interface_classdb_register_extension_class_virtual_method(
            gdextension_library, _class_name._native_ptr(), &mi
        )

        gdextension_interface_mem_free(def_args)
        gdextension_interface_mem_free(arguments_info)
        gdextension_interface_mem_free(arguments_metadata)
