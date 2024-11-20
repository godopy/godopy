cdef dict _NODEDB = {}


cdef class ExtensionClassBindings:
    def __cinit__(self):
        self.method = {}
        self.pymethod = {}
        self.gdvirtual = {}
        self.ownvirtual = {}
        self.intenum = {}
        self.bitfield = {}
        self.prop = {}
        self.idxprop = {}
        self.group = {}
        self.subgroup = {}
        self.signal = {}


cdef class ExtensionClass(Class):
    """
    Defines all custom classes which extend the Godot Engine.
    Inherits `gdextendion.Class`

    Implements all class registration/unregistration API calls:
        `classdb_register_extension_class4`
        `classdb_unregister_extension_class`

    Implements extension class callbacks in the ClassCreationInfo4 structure:
        `creation_info4.create_instance_func = &ExtensionClass.create_instance`
        `creation_info4.free_instance_func = &ExtensionClass.free_instance`
        `creation_info4.recreate_instance_func = &ExtensionClass.recreate_instance`
        `creation_info4.get_virtual_call_data_func = &ExtensionClass.get_virtual_call_data_callback`

    Stores information about all custom methods/properties/signals and class registration state.
    """
    def __init__(self, name: Str, object inherits: Str | Class, **kwargs):
        if not isinstance(inherits, (Class, str)):
            raise TypeError("'inherits' argument must be a Class instance or a string")

        self.__name__ = name
        if isinstance(inherits, Class):
            self.__inherits__ = inherits
        else:
            self.__inherits__ = Class.get_class(inherits)

        self.__method_info__ = {}

        self.is_registered = False
        self.is_virtual = kwargs.pop('is_virtual', False)
        self.is_abstract = kwargs.pop('is_abstract', False)
        self.is_exposed = kwargs.pop('is_exposed', True)
        self.is_runtime = kwargs.pop('is_runtime', False)

        self._bind = ExtensionClassBindings()

        self._used_refs = []


    def __call__(self):
        if not self.is_registered:
            raise RuntimeError("Extension class is not registered")

        return Extension(self, self.__inherits__)


    cdef object get_method_and_method_type_info(self, object name):
        cdef object method = self._bind.gdvirtual[name]
        cdef dict method_info = self.__inherits__.get_method_info(name)
        cdef tuple method_and_method_type_info = (method, method_info['type_info'])

        return method_and_method_type_info


    cdef void *get_method_and_method_type_info_ptr(self, object name) except NULL:
        cdef tuple method_and_method_type_info = self.get_method_and_method_type_info(name)

        self._used_refs.append(method_and_method_type_info)
        ref.Py_INCREF(method_and_method_type_info)

        return <void *><PyObject *>method_and_method_type_info


    cdef void *get_special_method_info_ptr(self, SpecialMethod method) except NULL:
        cdef tuple info = (method,)

        self._used_refs.append(info)
        ref.Py_INCREF(info)

        return <void *><PyObject *>info


    def bind_method(self, method: typing.Callable, name: Optional[Str] = None) -> typing.Callable:
        if not callable(method):
            raise ValueError("Callable is required, got %s" % type(method))
        name = name or method.__name__
        self._bind.method[name] = method

        return method

    cdef int register_method(self, func: typing.Callable, name: Str) except -1:
        cdef ExtensionMethod method = ExtensionMethod(func, name)

        return method.register(self)


    def bind_python_method(self, method: typing.Callable, name: Optional[Str] = None) -> typing.Callable:
        if not callable(method):
            raise ValueError("Callable is required, got %s" % type(method))
        name = name or method.__name__
        self._bind.pymethod[name] = method

        return method


    def bind_virtual_method(self, method: typing.Callable, name: Optional[Str] = None) -> typing.Callable:
        if not callable(method):
            raise ValueError("Callable is required, got %s" % type(method))
        name = name or method.__name__
        self._bind.gdvirtual[name] = method

        return method


    def add_virtual_method(self, method: typing.Callable, name: Optional[Str] = None) -> typing.Callable:
        if not callable(method):
            raise ValueError("Callable is required, got %s" % type(method))
        name = name or method.__name__
        self._bind.ownvirtual[name] = method

        return method


    cdef int register_virtual_method(self, func: typing.Callable, name: Str) except -1:
        cdef ExtensionVirtualMethod method = ExtensionVirtualMethod(func, name)

        return method.register(self)


    def bind_int_enum(self, enum_obj: enum.IntEnum) -> enum.IntEnum:
        self._bind.intenum[enum_obj.__name__] = enum_obj

        return enum_obj

    def bind_bitfield(self, enum_obj: enum.IntEnum) -> enum.IntEnum:
        self._bind.bitfield[enum_obj.__name__] = enum_obj

        return enum_obj


    cdef int register_enum(self, enum_obj: enum.IntEnum, bint is_bitfield=False) except -1:
        cdef ExtensionEnum ext_enum = ExtensionEnum(enum_obj, is_bitfield=is_bitfield)

        return ext_enum.register(self)


    def add_property(self, info: PropertyInfo, setter: types.FunctionType | Str,
                     getter: types.FunctionType | Str) -> None:
        cdef object setter_name, getter_name

        if not info.name:
            raise ValueError("Empty 'info.name'")

        if isinstance(setter, str):
            setter_name = setter
            try:
                setter = self._bind.method[setter_name]
            except KeyError:
                raise ValueError("Setter %r was not found" % setter_name)
        elif isinstance(setter, type.FunctionType):
            setter_name = setter.__name__
            if setter_name not in self._bind.method:
                self._bind.method[setter_name] = setter
        else:
            raise ValueError("Invalid setter argument %r" % setter)

        if isinstance(getter, str):
            getter_name = getter
            try:
                getter = self._bind.method[getter_name]
            except KeyError:
                raise ValueError("Getter %r was not found" % getter_name)
        elif isinstance(setter, type.FunctionType):
            getter_name = getter.__name__
            if getter_name not in self._bind.method:
                self._bind.method[getter_name] = getter
        else:
            raise ValueError("Invalid getter argument %r" % getter)

        self._bind.prop[info.name] = (info, setter_name, getter_name)


    cdef int register_property(self, PropertyInfo info, setter_name: Str, getter_name: Str) except -1:
        cdef ExtensionProperty prop = ExtensionProperty(info, setter_name, getter_name)

        return prop.register(self)


    def add_property_i(self, info: PropertyInfo, setter: types.FunctionType | Str,
                     getter: types.FunctionType | Str, int64_t idx) -> None:
        cdef object setter_name, getter_name

        if not info.name:
            raise ValueError("Empty 'info.name'")

        if isinstance(setter, str):
            setter_name = setter
            try:
                setter = self._bind.method[setter_name]
            except KeyError:
                raise ValueError("Setter %r was not found" % setter_name)
        elif isinstance(setter, type.FunctionType):
            setter_name = setter.__name__
            if setter_name not in self._bind.method:
                self._bind.method[setter_name] = setter
        else:
            raise ValueError("Invalid setter argument %r" % setter)

        if isinstance(getter, str):
            getter_name = getter
            try:
                getter = self._bind.method[getter_name]
            except KeyError:
                raise ValueError("Getter %r was not found" % getter_name)
        elif isinstance(setter, type.FunctionType):
            getter_name = getter.__name__
            if getter_name not in self._bind.method:
                self._bind.method[getter_name] = getter
        else:
            raise ValueError("Invalid getter argument %r" % getter)

        self._bind.idxprop[info.name] = (info, setter_name, getter_name, idx)

        return getter


    cdef int register_property_indexed(self, PropertyInfo info, setter_name: Str, getter_name: Str,
                                       int64_t index) except -1:
        cdef ExtensionProperty prop = ExtensionProperty(info, setter_name, getter_name, index)

        return prop.register(self)


    def add_group(self, group_name: Str, prefix: Str = '') -> None:
        self._bind.group[group_name] = (group_name, prefix)


    cdef int register_property_group(self, group_name: Str, prefix: Str) except -1:
        cdef ExtensionPropertyGroup grp = ExtensionPropertyGroup(group_name, prefix)

        return grp.register(self)


    def add_subgroup(self, subgroup_name: Str, prefix: Str = '') -> None:
        self._bind.subgroup[subgroup_name] = (subgroup_name, prefix)


    cdef int register_property_subgroup(self, subgroup_name: Str, prefix: Str) except -1:
        cdef ExtensionPropertyGroup grp = ExtensionPropertyGroup(subgroup_name, prefix, is_subgroup=True)

        return grp.register(self)


    def add_signal(self, signal_name: Str, arguments: Sequence[PropertyInfo] = None) -> None:
        self._bind.signal[signal_name] = arguments or []


    cdef int register_signal(self, signal_name: Str, arguments: Sequence[PropertyInfo]) except -1:
        cdef ExtensionSignal sig = ExtensionSignal(signal_name, arguments)

        return sig.register(self)


    cdef int set_registered(self) except -1:
        self.is_registered = True
        _CLASSDB[self.__name__] = self

        cdef PyGDStringName class_name = PyGDStringName(self.__name__)
        self._godot_class_tag = gdextension_interface_classdb_get_class_tag(class_name.ptr())

        return 0


    cdef int unregister(self) except -1:
        if not self.is_registered:
            return 0

        for reference in self._used_refs:
            # print(reference, Py_REFCNT(reference))
            ref.Py_DECREF(reference)

        self._used_refs = []

        cdef PyGDStringName class_name = PyGDStringName(self.__name__)
        gdextension_interface_classdb_unregister_extension_class(gdextension_library, class_name.ptr())

        # print(self, Py_REFCNT(self))
        ref.Py_DECREF(self)

        return 0


    def __del__(self):
        if self.is_registered:
            self.unregister()


    def register(self, **kwargs):
        try:
            return self._register(kwargs)
        except Exception as exc:
            print_traceback_and_die(exc)


    def register_abstract(self):
        return self.register(is_abstract=True)


    def register_internal(self):
        return self.register(is_exposed=False)


    def register_runtime(self):
        return self.register(is_runtime=True)


    @staticmethod
    cdef void *create_instance_callback(void *p_class_userdata, uint8_t p_notify_postinitialize) noexcept nogil:
        with gil:
            if p_class_userdata == NULL:
                UtilityFunctions.push_error("ExtensionClass object pointer is uninitialized")
            self = <object>p_class_userdata
            return (<ExtensionClass>self).create_instance(p_notify_postinitialize)


    cdef void *create_instance(self, bint p_notify_postinitialize) except? NULL:
        from entry_point import get_config
        config = get_config()
        godot_class_to_class = config.get('godot_class_to_class')

        cls = godot_class_to_class(self)
        cdef uint64_t self_id = <uint64_t><PyObject *>self
        cdef Extension instance = cls(__godot_class__=self, from_callback=True, _internal_check=hex(self_id))

        if self.issubclass('Node'):
            if self.__name__ in _NODEDB:
                UtilityFunctions.push_warning(
                    "%s instance already saved to _NODEDB: %r, but another instance %r was requested, rewriting"
                    % (self.__name__, _NODEDB[self.__name__], instance)
                )

                _NODEDB[self.__name__] = instance
            else:
                # print('Saved %r instance %r' % (self, instance))

                _NODEDB[self.__name__] = instance

        return instance._owner


    @staticmethod
    cdef void free_instance_callback(void *p_class_userdata, void *p_instance) noexcept nogil:
        with gil:
            self = <object>p_class_userdata
            (<ExtensionClass>self).free_instance(<object>p_instance)


    cdef int free_instance(self, object instance) except -1:
        if self.__name__ in _NODEDB:
            del _NODEDB[self.__name__]

        ref.Py_DECREF(instance)

        return 0


    @staticmethod
    cdef void *recreate_instance_callback(void *p_class_userdata, void *p_instance) noexcept nogil:
        with gil:
            print('ExtensinoClass recreate callback called; classs ptr:%x instance ptr:%x'
                  % (<uint64_t>p_class_userdata, <uint64_t>p_instance))
        return NULL


    @staticmethod
    cdef void *get_virtual_call_data_callback(void *p_class_userdata, GDExtensionConstStringNamePtr p_name) noexcept nogil:
        cdef StringName name = deref(<StringName *>p_name)

        # Ensure that PyThreadState is created for the current Godot thread,
        # otherwise calling a GIL function from uninitialized threads would create a deadlock
        PythonRuntime.get_singleton().ensure_current_thread_state()

        cdef void *ret

        with gil:
            self = <object>p_class_userdata
            (<ExtensionClass>self).get_virtual_call_data(type_funcs.string_name_to_pyobject(name), &ret)

            return ret


    cdef int get_virtual_call_data(self, object name, void **r_ret) except -1:
        cdef void* func_and_typeinfo_ptr

        # Special case, some virtual methods of ScriptLanguageExtension
        # which does not belong to Python ScriptLanguage implementations
        if name in special_method_to_enum:
            r_ret[0] = self.get_special_method_info_ptr(special_method_to_enum[name])
            return 0

        if name not in self._bind.gdvirtual:
            r_ret[0] = NULL
            return 0

        r_ret[0] = self.get_method_and_method_type_info_ptr(name)


    cdef int _register(self, object kwargs) except -1:
        if self.is_registered:
            raise RuntimeError("%r is already registered" % self)

        cdef GDExtensionClassCreationInfo4 ci
        cdef void *self_ptr = <PyObject *>self

        ci.is_virtual = kwargs.pop('is_virtual', self.is_virtual)
        ci.is_abstract = kwargs.pop('is_abstract', self.is_abstract)
        ci.is_exposed = kwargs.pop('is_exposed', self.is_exposed)
        ci.is_runtime = kwargs.pop('is_runtime', self.is_runtime)

        self.is_virtual = ci.is_virtual
        self.is_abstract = ci.is_abstract
        self.is_exposed = ci.is_exposed
        self.is_runtime = ci.is_runtime

        ci.set_func = &Extension.set_callback
        ci.get_func = &Extension.get_callback

        if kwargs.pop('has_get_property_list', False):
            ci.get_property_list_func = &Extension.get_property_list_callback
        else:
            ci.get_property_list_func = NULL

        ci.free_property_list_func = NULL
        ci.property_can_revert_func = NULL
        ci.property_get_revert_func = NULL
        ci.validate_property_func = NULL
        ci.notification_func = &Extension.notification_callback
        ci.to_string_func = &Extension.to_string_callback
        ci.reference_func = NULL
        ci.unreference_func = NULL
        ci.create_instance_func = &ExtensionClass.create_instance_callback
        ci.free_instance_func = &ExtensionClass.free_instance_callback
        ci.recreate_instance_func = &ExtensionClass.recreate_instance_callback

        ci.get_virtual_func = NULL
        ci.get_virtual_call_data_func = &ExtensionClass.get_virtual_call_data_callback
        ci.call_virtual_with_data_func = &Extension.call_virtual_with_data_callback
        ci.class_userdata = self_ptr

        ref.Py_INCREF(self) # DECREF in ExtensionClass.unregister()

        cdef StringName class_name, inherits_name
        type_funcs.string_name_from_pyobject(self.__name__, &class_name)
        type_funcs.string_name_from_pyobject(self.__inherits__.__name__, &inherits_name)

        gdextension_interface_classdb_register_extension_class4(
            gdextension_library,
            class_name._native_ptr(),
            inherits_name._native_ptr(),
            &ci
        )

        for name, method in self._bind.method.items():
            self.register_method(method, name)

        for name, method in self._bind.ownvirtual.items():
            self.register_virtual_method(method, name)

        for enum in self._bind.intenum.values():
            self.register_enum(enum)

        for enum in self._bind.bitfield.values():
            self.register_enum(enum, True)

        for info, setter_name, getter_name in self._bind.prop.values():
            self.register_property(info, setter_name, getter_name)

        for info, setter_name, getter_name, idx in self._bind.idxprop.values():
            self.register_property_indexed(info, setter_name, getter_name, idx)

        for name, prefix in self._bind.group.values():
            self.register_property_group(name, prefix)

        for name, prefix in self._bind.subgroup.values():
            self.register_property_subgroup(name, prefix)

        for name, arguments in self._bind.signal.items():
            self.register_signal(name, arguments)

        self.set_registered()

        # print("%r is registered\n" % self)
