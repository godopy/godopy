cdef class PropertyInfo:
    def __cinit__(self, VariantType type, str name='', str class_name='', uint32_t hint=0, str hint_string='', uint32_t usage=0):
        self.type = type
        self.name = name
        self.class_name = class_name
        self.hint = hint
        self.hint_string = hint_string
        self.usage = usage

    def __repr__(self):
        cls_name = '%s.' % self.class_name if self.class_name else ''
        name = self.name or '<return-value>'

        return '<PropertyInfo %s%s:%s>' % (cls_name, name, variant_type_to_str(self.type))


@cython.final
cdef class _GDEPropInfoData:
    cdef _Memory memory
    cdef StringName name
    cdef StringName classname
    cdef StringName hintstring

    def __cinit__(self, PropertyInfo propinfo):
        self.memory = _Memory(cython.sizeof(GDExtensionPropertyInfo))
        self.name = StringName(<const PyObject *>propinfo.name)
        self.classname = StringName(<const PyObject *>propinfo.class_name)
        self.hintstring = StringName(<const PyObject *>propinfo.hint_string)

        (<GDExtensionPropertyInfo *>self.memory.ptr).type = <GDExtensionVariantType>(<VariantType>propinfo.type)
        (<GDExtensionPropertyInfo *>self.memory.ptr).name = self.name._native_ptr()
        (<GDExtensionPropertyInfo *>self.memory.ptr).class_name = self.classname._native_ptr()
        (<GDExtensionPropertyInfo *>self.memory.ptr).hint = propinfo.hint
        (<GDExtensionPropertyInfo *>self.memory.ptr).hint_string = self.hintstring._native_ptr()
        (<GDExtensionPropertyInfo *>self.memory.ptr).usage = propinfo.usage

    def __dealloc__(self):
        self.memory.free()

    cdef GDExtensionPropertyInfo *ptr(self):
        return <GDExtensionPropertyInfo *>self.memory.ptr


@cython.final
cdef class _GDEPropInfoListData:
    cdef _Memory memory
    cdef size_t count
    cdef list names
    cdef list classnames
    cdef list hintstrings

    def __cinit__(self, object propinfo_list):
        cdef size_t i
        self.count = len(propinfo_list)
        self.memory = _Memory(cython.sizeof(GDExtensionPropertyInfo) * self.count)

        self.names = [PyStringName(prop_info.name) for prop_info in propinfo_list]
        self.classnames = [PyStringName(prop_info.class_name) for prop_info in propinfo_list]
        self.hintstrings = [PyStringName(prop_info.hint_string) for prop_info in propinfo_list]

        cdef PropertyInfo propinfo

        for i in range(self.count):
            propinfo = propinfo_list[i]
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].type = \
                <GDExtensionVariantType>(<VariantType>propinfo.type)
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].name = \
                (<PyStringName>self.names[i]).ptr()
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].class_name = \
                (<PyStringName>self.classnames[i]).ptr()
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].hint = propinfo.hint
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].hint_string = \
                (<PyStringName>self.hintstrings[i]).ptr()
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].usage = propinfo.usage

    def __dealloc__(self):
        self.memory.free()

    cdef GDExtensionPropertyInfo *ptr(self):
        return <GDExtensionPropertyInfo *>self.memory.ptr


@cython.final
cdef class _GDEArgumentMetadataArray:
    cdef _Memory memory
    cdef size_t count

    def __cinit__(self, object argmeta_list):
        cdef size_t i
        self.count = len(argmeta_list)
        self.memory = _Memory(cython.sizeof(int) * self.count)

        for i in range(self.count):
            (<int *>self.memory.ptr)[i] = <int>argmeta_list[i]

    def __dealloc__(self):
        self.memory.free()

    cdef GDExtensionClassMethodArgumentMetadata *ptr(self):
        return <GDExtensionClassMethodArgumentMetadata *>self.memory.ptr


cdef class _ExtensionMethodBase:
    def __cinit__(self, *args):
        self.is_registered = False
        self.type_info = ()
        self.__func__ = None
        self.__name__ = ''


    def __init__(self, object method: types.FunctionType):
        self.__func__ = method
        self.__name__ = method.__name__


    def __str__(self):
        return self.__name__


    def __repr__(self):
        cdef str args = "(%s)" % ', '.join(ai.name for ai in self.get_argument_info_list()[1:])
        cdef str qualifier = 'Unbound'
        if not self.is_registered:
            qualifier += ' unregistered'
        return "<%s %s %s%s>" % (qualifier, self.__class__.__name__, self.__name__, args)


    cdef list get_default_arguments(self):
        if self.__func__.__defaults__ is None:
            return []
        return [arg for arg in self.__func__.__defaults__]


    cdef PropertyInfo get_argument_info(self, int pos):
        cdef PropertyInfo pi = PropertyInfo(NIL)

        if pos >= 0:
            pi.name = self.__func__.__code__.co_varnames[pos]
            pi.type = type_funcs.pytype_to_variant_type(self.__func__.__annotations__.get(pi.name, None))

        return pi


    cdef PropertyInfo get_return_info(self):
        return PropertyInfo(
            type_funcs.pytype_to_variant_type(self.__func__.__annotations__.get('return', None)),
        )


    cdef list get_argument_info_list(self):
        return [self.get_argument_info(i) for i in range(self.get_argument_count())]


    cdef int get_return_metadata(self) noexcept:
        cdef VariantType t = type_funcs.pytype_to_variant_type(self.__func__.__annotations__.get('return', None))

        return self.metadata_from_type(t)


    cdef int metadata_from_type(self, VariantType t) noexcept nogil:
        if t == INT:
            return <int>GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT64
        elif t == FLOAT:
            return <int>GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_DOUBLE

        return <int>GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE


    cdef list get_argument_metadata_list(self):
        cdef size_t i
        cdef list metadata_list = []
        cdef int metadata
        cdef VariantType t
        for i in range(self.get_argument_count()):
            metadata = <int>GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE
            if i > 0:
                name = self.__func__.__code__.co_varnames[i]
                t = type_funcs.pytype_to_variant_type(self.__func__.__annotations__.get(name, None))
                metadata = self.metadata_from_type(t)
            metadata_list.append(metadata)

        return metadata_list


    cdef GDExtensionBool has_return(self) noexcept:
        return <GDExtensionBool>bool(self.__func__.__annotations__.get('return'))


    cdef uint32_t get_argument_count(self) noexcept:
        # includes self
        return <uint32_t>self.__func__.__code__.co_argcount
