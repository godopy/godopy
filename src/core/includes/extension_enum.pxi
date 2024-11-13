cdef class ExtensionEnum:
    """"
    Defines all integer enumerations and bitfields of `gdextension.Extension` objects.

    Implements following GDExtension API calls:
        in `ExtensionEnum.register`
            `classdb_register_extension_class_integer_constant`
    """
    def __cinit__(self, *args, **kwargs):
        self.__name__ = ''
        self.__enum__ = None
        self.is_registered = False
        self.is_bitfield = False


    def __init__(self, enum: enum.IntEnum, bint is_bitfield=False) -> None:
        self.__enum__ = enum
        self.__name__ = enum.__name__
        self.is_bitfield = is_bitfield


    cdef int register(self, ExtensionClass cls) except -1:
        cdef PyGDStringName class_name = PyGDStringName(cls.__name__)
        cdef PyGDStringName enum_name = PyGDStringName(self.__name__)
        cdef PyGDStringName constant_name

        for field in self.__enum__:
            constant_name = PyGDStringName(field.name)

            gdextension_interface_classdb_register_extension_class_integer_constant(
                gdextension_library,
                class_name.ptr(),
                enum_name.ptr(),
                constant_name.ptr(),
                field.value,
                self.is_bitfield
            )

        self.is_registered = True
