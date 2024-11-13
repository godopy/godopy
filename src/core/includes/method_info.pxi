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

    def as_dict(self):
        return {
            'name': self.name,
            'type': self.type,
            # 'class_name': self.class_name,
            # 'hint': self.hint,
            # 'hint_string': self.hint_string,
            # 'usage': self.usage
        }


cdef class MethodInfo:
    def __cinit__(self, name: Str, object arguments, int32_t id, PropertyInfo return_value=None, uint32_t flags=0,
                  object default_arguments=None) -> None:
        self.name = str(name)
        if return_value is None:
            self.return_value = PropertyInfo(NIL)
        else:
            self.return_value = return_value
        if flags == 0:
            self.flags = GDEXTENSION_METHOD_FLAG_NORMAL
        else:
            self.flags = flags
        self.id = id

        self.arguments = arguments
        self.default_arguments = default_arguments or []

    def __repr__(self):
        arguments = ', '.join(f'{arg.name}: {variant_type_to_str(arg.type)}' for arg in self.arguments)
        ret = variant_type_to_str(self.return_value.type)
        return f'<MethodInfo {self.name}({arguments}) -> {ret}'

    def as_dict(self):
        return {
            'name': self.name,
            'return_value': self.return_value.as_dict(),
            'id': self.id,
            'flags': self.flags,
            'arguments': [arg.as_dict() for arg in self.arguments],
            'default_arguments': []
        }


@cython.final
cdef class _PropertyInfoData:
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
cdef class _PropertyInfoDataArray:
    def __cinit__(self, object propinfo_list):
        self.alloc(propinfo_list)

    def alloc(self, object propinfo_list):
        cdef size_t i
        self.count = len(propinfo_list)
        self.memory = _Memory(cython.sizeof(GDExtensionPropertyInfo) * self.count)

        self.names = [PyGDStringName(prop_info.name) for prop_info in propinfo_list]
        self.classnames = [PyGDStringName(prop_info.class_name) for prop_info in propinfo_list]
        self.hintstrings = [PyGDStringName(prop_info.hint_string) for prop_info in propinfo_list]

        cdef PropertyInfo propinfo

        for i in range(self.count):
            propinfo = propinfo_list[i]
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].type = <GDExtensionVariantType>(<VariantType>propinfo.type)
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].name = (<PyGDStringName>self.names[i]).ptr()
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].class_name = (<PyGDStringName>self.classnames[i]).ptr()
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].hint = propinfo.hint
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].hint_string = (<PyGDStringName>self.hintstrings[i]).ptr()
            (<GDExtensionPropertyInfo *>self.memory.ptr)[i].usage = propinfo.usage

    def free(self):
        self.memory.free()
        self.names = []
        self.classnames = []
        self.hintstring = []
        self.count = 0

    def __dealloc__(self):
        self.memory.free()

    cdef GDExtensionPropertyInfo *ptr(self) noexcept nogil:
        return <GDExtensionPropertyInfo *>self.memory.ptr


@cython.final
cdef class _MethodInfoDataArray:
    def __cinit__(self, object methodinfo_list):
        self.alloc(methodinfo_list)

    def alloc(self, object methodinfo_list):
        cdef size_t i
        self.count = len(methodinfo_list)
        self.memory = _Memory(cython.sizeof(GDExtensionMethodInfo) * self.count)

        self.names = [PyGDStringName(info.name) for info in methodinfo_list]

        self.return_values = _PropertyInfoDataArray([info.return_value for info in methodinfo_list])
        self.arguments = [_PropertyInfoDataArray(info.arguments) for info in methodinfo_list]

        cdef MethodInfo methodinfo

        for i in range(self.count):
            methodinfo = methodinfo_list[i]
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].name = (<PyGDStringName>self.names[i]).ptr()
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].return_value = \
                (<GDExtensionPropertyInfo *>self.return_values.ptr)[i]
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].flags = methodinfo.flags
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].id = methodinfo.id
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].argument_count = len(methodinfo.arguments)
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].arguments = \
                (<GDExtensionPropertyInfo *>self.arguments[i].memory.ptr)
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].default_argument_count = 0
            (<GDExtensionMethodInfo *>self.memory.ptr)[i].default_arguments = NULL

    def __dealloc__(self):
        self.memory.free()
        self.return_values.free()
        for arg in self.arguments:
            arg.free()

    cdef GDExtensionMethodInfo *ptr(self) noexcept nogil:
        return <GDExtensionMethodInfo *>self.memory.ptr


@cython.final
cdef class _ArgumentMetadataArray:
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
