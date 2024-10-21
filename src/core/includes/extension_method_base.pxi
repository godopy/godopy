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
        return '<PropertyInfo %s%s:%s>' % (cls_name, self.name, variant_type_to_str(self.type))


cdef class PyStringName:
    cdef str pyname
    cdef StringName name

    def __cinit__(self, str name):
        self.pyname = name
        self.name = StringName(name)

    cdef void *ptr(self):
        return self.name._native_ptr()


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
            pi.type = pytype_to_gdtype(self.__func__.__annotations__.get(pi.name, None))

        return pi


    cdef PropertyInfo get_return_info(self):
        return PropertyInfo(
            pytype_to_gdtype(self.__func__.__annotations__.get('return', None)),
        )


    cdef list get_argument_info_list(self):
        return [self.get_argument_info(i) for i in range(self.get_argument_count())]


    cdef int get_return_metadata(self) except -1:
        cdef VariantType t = pytype_to_gdtype(self.__func__.__annotations__.get('return', None))
        return self.metadata_from_type(t)


    cdef int metadata_from_type(self, VariantType t) except -1 nogil:
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
                t = pytype_to_gdtype(self.__func__.__annotations__.get(name, None))
                metadata = self.metadata_from_type(t)
            metadata_list.append(metadata)

        return metadata_list


    cdef GDExtensionBool has_return(self) except -1:
        return <GDExtensionBool>bool(self.__func__.__annotations__.get('return'))


    cdef uint32_t get_argument_count(self) except -1:
        # includes self
        return <uint32_t>self.__func__.__code__.co_argcount


    cdef uint32_t get_default_argument_count(self) except -1:
        if self.__func__.__defaults__ is None:
            return 0
        return <uint32_t>len(self.__func__.__defaults__)
