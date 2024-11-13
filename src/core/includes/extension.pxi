cdef dict special_method_to_enum = {
    '_thread_enter': _THREAD_ENTER,
    '_thread_exit': _THREAD_EXIT,
    '_frame': _FRAME
}


cdef class Extension(Object):
    """
    Defines all `gdextension.Extension` instances defined by `gdextension.ExtensionClass`.

    Inherits `gdextension.Object` and all its methods.

    Implements following GDExtension API calls:
        in `Extension.__init__`
            `classdb_construct_object2`
            `object_set_instance`

    Implements virtual call callback in the ClassCreationInfo4 structure:
        `creation_info4.call_virtual_with_data_func = &Extension.call_virtual_with_data_callback`
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

        self._callable_cache = {}

        # print("%r initialized, from callback: %r" % (self, from_callback))


    @staticmethod
    cdef uint8_t set_callback(void *p_instance, const void *p_name, const void *p_value) noexcept nogil:
        cdef Variant value = deref(<Variant *>p_value)

        with gil:
            self = <object>p_instance
            try:
                setattr(
                    self,
                    type_funcs.string_name_to_pyobject(deref(<StringName *>p_name)),
                    type_funcs.variant_to_pyobject(value)
                )

                return True

            except AttributeError:
                return False


    @staticmethod
    cdef uint8_t get_callback(void *p_instance, const void *p_name, void *r_ret) noexcept nogil:
        with gil:
            self = <object>p_instance
            try:
                ret = getattr(self, type_funcs.string_name_to_pyobject(deref(<StringName *>p_name)))
                type_funcs.variant_from_pyobject(ret, <Variant *>r_ret)

                return True
            except AttributeError:
                return False


    @staticmethod
    cdef const GDExtensionPropertyInfo *get_property_list_callback(void *p_instance, uint32_t *r_count) noexcept nogil:
        with gil:
            UtilityFunctions.print("Extension.get_property_list_callback")
            self = <object>p_instance
            try:
                property_list = self.get_property_list()
                (<Extension>self).property_info_data = _PropertyInfoDataArray(property_list)
                if r_count:
                    r_count[0] = <uint32_t>((<Extension>self).property_info_data).count
                return <const GDExtensionPropertyInfo *>((<Extension>self).property_info_data).ptr

            except Exception as exc:
                print_error_with_traceback(exc)

                if r_count:
                    r_count[0] = <uint32_t>0
                return NULL


    def get_property_list(self) -> List[PropertyInfo]:
        cdef list propinfo_list = []
        cdef VariantType vartype
        cdef uint32_t hint = 0, usage = 0

        if not hasattr(self, '__dict__'):
            return propinfo_list

        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue

            vartype = type_funcs.pytype_to_variant_type(type(value))
            if hasattr(value, '__hint__'):
                hint = value.__hint__
            if hasattr(value, '__usage__'):
                usage = usage

            propinfo_list.append(PropertyInfo(vartype, key, self.__name__, hint=hint, usage=usage))

        return propinfo_list


    @staticmethod
    cdef void notification_callback(void *p_instance, int32_t p_what, uint8_t p_reversed) noexcept nogil:
        with gil:
            # UtilityFunctions.print("Extension.notification_callback")
            self = <object>p_instance
            try:
                # UtilityFunctions.print("Extension.notification_callback %r" % p_what)
                self.notification(p_what, p_reversed)
            except Exception as exc:
                print_error_with_traceback(exc)


    def notification(self, what: int, reversed: bool) -> None:
        pass


    @staticmethod
    cdef void to_string_callback(void *p_instance, uint8_t *r_is_valid, void *r_out) noexcept nogil:
        with gil:
            UtilityFunctions.print("Extension.to_string_callback")
            self = <object>p_instance
            try:
                type_funcs.string_from_pyobject(repr(self), <String *>r_out)
                type_funcs.bool_from_pyobject(True, r_is_valid)
            except Exception as exc:
                print_error_with_traceback(exc)


    @staticmethod
    cdef void call_virtual_with_data_callback(void *p_instance, const void *p_name, void *p_func,
                                              const (const void *) *p_args, void *r_ret) noexcept nogil:
        with gil:
            self = <object>p_instance
            (<Extension>self).call_virtual_with_data(<object>p_func, <const void **>p_args, r_ret)


    cdef void call_special_virtual(self, SpecialMethod method) noexcept nogil:
        if method == _THREAD_ENTER:
            # Create PyThreadState for every Godot thread
            PythonRuntime.get_singleton().ensure_current_thread_state()

        elif method == _THREAD_EXIT:
            # PyThreadStates are destroyed on exit, do nothing here
            pass

        elif method == _FRAME:
            # Called every frame, might be useful later
            pass


    cdef int call_virtual_with_data(self, object func_and_info, const void **p_args, void *r_ret) except -1:
        cdef object func = func_and_info[0]
        cdef SpecialMethod _special_func

        if PyLong_Check(func):
            _special_func = <SpecialMethod>PyLong_AsSsize_t(func)
            with nogil:
                self.call_special_virtual(_special_func)
            return 0

        cdef PythonCallable method
        cdef uint64_t key = <uint64_t><PyObject *>func

        if key in self._callable_cache:
            method = self._callable_cache[key]
        else:
            method = PythonCallable(self, func, func_and_info[1])
            self._callable_cache[key] = method

        try:
            _make_python_ptrcall(method, r_ret, p_args, method.get_argument_count())
        except Exception as exc:
            method.error_count += 1
            if method.error_count > 1:
                print_traceback_and_die(exc)
            else:
                print_error_with_traceback(exc)
