class ClassDB:
    @staticmethod
    def construct_object(classname: Str) -> Object:
        """
        Constructs an Object of the requested class.
        """
        return Object(classname)


    @staticmethod
    def get_method_bind(classname: Str, methodname: Str) -> MethodBind:
        """
        Gets a pointer to the MethodBind in ClassDB for the given class and method.
        """
        cdef Object obj = Object(classname)

        return MethodBind(obj, methodname)


    @staticmethod
    def get_class_tag(classname: Str) -> int:
        """
        Gets an integer ID uniquely identifying the given built-in class in the Godot's ClassDB.
        """
        cdef Class cls = Class.get_class(classname)

        return <uint64_t>cls._godot_class_tag


    @staticmethod
    def register_extension_class(class_name: Str, parent_class_name: Str, **kwargs) -> ExtensionClass:
        """
        Registers an extension class in the ClassDB.
        """
        cdef ExtensionClass cls = ExtensionClass(class_name, parent_class_name)

        cls.register(**kwargs)

        return cls


    @staticmethod
    def register_extension_class_method(ExtensionClass cls, method: typing.Callable) -> None:
        """
        Registers a method on an extension class in the ClassDB.
        """
        cls.register_method(method, method.__name__)


    @staticmethod
    def register_extension_class_virtual_method(ExtensionClass cls, method: typing.Callable) -> None:
        """
        Registers a virtual method on an extension class in ClassDB, that can be implemented by scripts
        or other extensions.
        """
        cls.register_virtual_method(method, method.__name__)


    @staticmethod
    def register_extension_class_integer_enum(ExtensionClass cls, enum_object: enum.IntEnum,
                                              bint is_bitfield=False) -> None:
        """
        Registers an integer enumeration on an extension class in the ClassDB.
        """
        cls.register_enum(enum_object, is_bitfield=is_bitfield)


    @staticmethod
    def register_extension_class_property(ExtensionClass cls, PropertyInfo info, setter: Str, getter: Str) -> None:
        """
        Registers a property on an extension class in the ClassDB.
        """
        cls.register_property(info, setter, getter)


    @staticmethod
    def register_extension_class_property_indexed(ExtensionClass cls, PropertyInfo info, setter: Str, getter: Str,
                                                  int64_t index) -> None:
        """
        Registers an indexed property on an extension class in the ClassDB.
        """
        cls.register_property_indexed(info, setter, getter, index)


    @staticmethod
    def register_extension_class_property_group(ExtensionClass cls, group_name: Str, prefix: Str) -> None:
        """
        Registers a property group on an extension class in the ClassDB.
        """
        cls.register_property_group(group_name, prefix)


    @staticmethod
    def register_extension_class_property_subgroup(ExtensionClass cls, subgroup_name: Str, prefix: Str) -> None:
        """
        Registers a property subgroup on an extension class in the ClassDB.
        """
        cls.register_property_subgroup(subgroup_name, prefix)


    @staticmethod
    def register_extension_class_signal(ExtensionClass cls, signal_name: Str, attributes: List[PropertyInfo]) -> None:
        """
        Registers a signal on an extension class in the ClassDB.
        """
        cls.register_property_signal(signal_name, attributes)


    @staticmethod
    def unregister_extension_class(ExtensionClass cls) -> None:
        """
        Unregisters an extension class in the ClassDB.
        """
        del cls
