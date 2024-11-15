cdef class ExtensionSignal:
    """"
    Defines all custom signals of `gdextension.Extension` objects.

    Implements following GDExtension API calls:
        in `ExtensionSignal.register`
            `classdb_register_extension_class_signal`
    """
    def __cinit__(self, *args, **kwargs):
        self.__name__ = ''
        self.__arguments__ = []
        self.is_registered = False


    def __init__(self, name: Str, arguments: Sequence[PropertyInfo] = None) -> None:
        self.__name__ = name
        if arguments is not None:
            self.__arguments__ = list(arguments)


    cdef int register(self, ExtensionClass cls) except -1:
        cdef PyGDStringName class_name = PyGDStringName(cls.__name__)
        cdef PyGDStringName signal_name = PyGDStringName(self.__name__)
        cdef _PropertyInfoDataArray data

        if not self.__arguments__:
            gdextension_interface_classdb_register_extension_class_signal(
                gdextension_library,
                class_name.ptr(),
                signal_name.ptr(),
                NULL,
                0
            )

        else:
            data = _PropertyInfoDataArray(self.__arguments__)

            gdextension_interface_classdb_register_extension_class_signal(
                gdextension_library,
                class_name.ptr(),
                signal_name.ptr(),
                data.ptr(),
                data.count
            )

        self.is_registered = True
