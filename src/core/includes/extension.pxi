cdef dict special_method_to_enum = {
    '_thread_enter': _THREAD_ENTER,
    '_thread_exit': _THREAD_EXIT,
    '_frame': _FRAME
}


cdef dict _bound_method_cache = {}


cdef class Extension(Object):
    """
    Defines all instances of `gdextension.ExtensionClass`.

    Inherits `gdextension.Object` and all its methods.

    Implements following GDExtension API calls:
        in `Extension.__init__`
            `classdb_construct_object2`
            `object_set_instance`

    Implements virtual call callbacks in the ClassCreationInfo4 structure:
        `creation_info4.get_virtual_call_data_func = &Extension.get_virtual_call_data`
        `creation_info4.call_virtual_with_data_func = &Extension.call_virtual_with_data`
    """
    def __init__(self, ExtensionClass cls=None, **kwargs):
        if cls is None:
            cls = kwargs.pop('__godot_class__', None)

        if not isinstance(cls, ExtensionClass):
            raise TypeError("Expected ExtensionClass instance, got %r" % type(cls))

        base_class = cls.__inherits__

        if not cls.is_registered:
            raise RuntimeError('Extension class must be registered')

        cdef bint from_callback = kwargs.pop('from_callback', False)

        self._needs_cleanup = not from_callback

        self.__godot_class__ = cls
        cdef PyGDStringName class_name = PyGDStringName(cls.__name__)
        cdef PyGDStringName base_class_name = PyGDStringName(base_class.__name__)

        self._owner = gdextension_interface_classdb_construct_object2(base_class_name.ptr())

        # INCREF because we lend a references of 'self' to the Godot Engine
        ref.Py_INCREF(self) # for set_instance, DECREF in ExtensionClass._free_instance

        cdef void *self_ptr = <void *><PyObject *>self

        gdextension_interface_object_set_instance(self._owner, class_name.ptr(), self_ptr)

        cdef object inner_init = self.__godot_class__._bind.pymethod.get('__inner_init__')

        if inner_init:
            inner_init(self)

        notification = MethodBind(self, 'notification')
        notification(0, False) # NOTIFICATION_POSTINITIALIZE

        _OBJECTDB[self.owner_id()] = self
        gdtypes.add_object_type(self.__class__)

        # print("%r initialized, from callback: %r" % (self, from_callback))


    @staticmethod
    cdef void *get_virtual_call_data(void *p_userdata, GDExtensionConstStringNamePtr p_name) noexcept nogil:
        cdef StringName name = deref(<StringName *>p_name)

        # Ensure that PyThreadState is created for the current Godot thread,
        # otherwise calling a GIL function from uninitialized threads would create a deadlock
        PythonRuntime.get_singleton().ensure_current_thread_state()

        return Extension._get_virtual_call_data(p_userdata, name)

    @staticmethod
    cdef void *_get_virtual_call_data(void *p_cls, const StringName &p_name) noexcept with gil:
        cdef ExtensionClass cls = <ExtensionClass>p_cls
        cdef str name = str(type_funcs.string_name_to_pyobject(p_name))

        cdef void* func_and_typeinfo_ptr

        # Special case, some virtual methods of ScriptLanguageExtension
        # which does not belong to Python ScriptLanguage implementations
        if name in special_method_to_enum:
            func_and_typeinfo_ptr = cls.get_special_method_info_ptr(special_method_to_enum[name])

            return func_and_typeinfo_ptr

        if name not in cls._bind.gdvirtual:
            return NULL

        func_and_typeinfo_ptr = cls.get_method_and_method_type_info_ptr(name)

        return func_and_typeinfo_ptr

    @staticmethod
    cdef void _call_special_virtual(SpecialMethod method) noexcept nogil:
        if method == _THREAD_ENTER:
            # Create PyThreadState for every Godot thread
            PythonRuntime.get_singleton().ensure_current_thread_state()

        elif method == _THREAD_EXIT:
            # PyThreadStates are destroyed on exit, do nothing here
            pass

        elif method == _FRAME:
            # Called every frame, might be useful later
            pass

    @staticmethod
    cdef void call_virtual_with_data(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name,
                                     void *p_func, const GDExtensionConstTypePtr *p_args,
                                     GDExtensionTypePtr r_ret) noexcept nogil:
        Extension._call_virtual_with_data(p_instance, p_func, <const void **>p_args, <void *>r_ret)


    @staticmethod
    cdef void _call_virtual_with_data(void *p_self, void *p_func_and_info, const void **p_args,
                                      void *r_ret) noexcept with gil:
        cdef tuple func_and_info = <tuple>p_func_and_info
        cdef object func = func_and_info[0]
        cdef SpecialMethod _special_func

        if PyLong_Check(func):
            _special_func = <SpecialMethod>PyLong_AsSsize_t(func)
            with nogil:
                Extension._call_special_virtual(_special_func)
            return

        cdef Extension self
        cdef PythonCallable method

        cdef uint32_t key = hash_murmur3_one_64(<uint64_t>p_self)
        key = hash_murmur3_one_64(<uint64_t><PyObject *>func, key)

        if key in _bound_method_cache:
            self, method = _bound_method_cache[key]
        else:
            self = <object>p_self
            method = PythonCallable(self, func, func_and_info[1])
            _bound_method_cache[key] = self, method

        try:
            _make_python_ptrcall(method, r_ret, p_args, method.get_argument_count())
        except Exception as exc:
            method.error_count += 1
            if method.error_count > 1:
                print_traceback_and_die(exc)
            else:
                print_error_with_traceback(exc)
