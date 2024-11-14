cdef class ExtensionProperty:
    """"
    Defines all properties of `gdextension.Extension` objects.

    Implements following GDExtension API calls:
        in `ExtensionProperty.register`
            `classdb_register_extension_class_property`
            `classdb_register_extension_class_property_indexed`
    """
    def __cinit__(self, *args):
        self.__name__ = ''
        self.__setter_name__ = ''
        self.__getter_name__ = ''
        self.__info__ = None
        self.is_registered = False
        self.is_indexed = False
        self.__idx__ = 0


    def __init__(self, PropertyInfo info, setter_name: Str, getter_name: Str, int64_t index=-1) -> None:
        if not info.name:
            raise ValueError("Empty 'info.name'")

        (<PropertyInfo>info).usage |= 4  # PROPERTY_USAGE_EDITOR

        self.__info__ = info
        self.__name__ = info.name
        self.__setter_name__ = setter_name
        self.__getter_name__ = getter_name

        if index > 0:
            self.is_indexed = True
            self.__idx__ = index


    cdef int register(self, ExtensionClass cls) except -1:
        cdef PyGDStringName class_name = PyGDStringName(cls.__name__)
        cdef PyGDStringName name = PyGDStringName(self.__name__)
        cdef PyGDStringName setter_name = PyGDStringName(self.__setter_name__)
        cdef PyGDStringName getter_name = PyGDStringName(self.__getter_name__)

        cdef _PropertyInfoData data = _PropertyInfoData(self.__info__)

        if self.is_indexed:
            gdextension_interface_classdb_register_extension_class_property_indexed(
                gdextension_library,
                class_name.ptr(),
                data.ptr(),
                setter_name.ptr(),
                getter_name.ptr(),
                self.__idx__
            )
        else:
            gdextension_interface_classdb_register_extension_class_property(
                gdextension_library,
                class_name.ptr(),
                data.ptr(),
                setter_name.ptr(),
                getter_name.ptr()
            )

        self.is_registered = True


cdef class ExtensionPropertyGroup:
    """"
    Defines all property groups/subgroups of `gdextension.Extension` objects.

    Implements following GDExtension API calls:
        in `ExtensionPropertyGroup.register`
            `classdb_register_extension_class_property_group`
            `classdb_register_extension_class_property_subgroup`
    """
    def __cinit__(self, *args, **kwargs):
        self.__name__ = ''
        self.__prefix__ = ''
        self.is_registered = False
        self.is_subgroup = False


    def __init__(self, name: Str, prefix: Str = '', bint is_subgroup=False):
        self.__name__ = name
        self.__prefix__ = prefix
        self.is_subgroup = is_subgroup


    cdef int register(self, ExtensionClass cls) except -1:
        cdef PyGDStringName class_name = PyGDStringName(cls.__name__)
        cdef PyGDStringName group_name = PyGDStringName(self.__name__)
        cdef PyGDStringName prefix = PyGDStringName(self.__prefix__)

        if self.is_subgroup:
            gdextension_interface_classdb_register_extension_class_property_subgroup(
                gdextension_library,
                class_name.ptr(),
                group_name.ptr(),
                prefix.ptr()
            )
        else:
            gdextension_interface_classdb_register_extension_class_property_group(
                gdextension_library,
                class_name.ptr(),
                group_name.ptr(),
                prefix.ptr()
            )

        self.is_registered = True
