cdef class ScriptInstance:
    def __init__(self, Extension script, Object owner, object cls_dict):
        self.__script__ = script
        self.__owner__ = owner
        self.__script_dict__ = {}
        self.__script_dict__.update(cls_dict)

        if hasattr(script, '_get_global_name'):
            self.__name__ = script._get_global_name()
        elif hasattr(script, 'get_global_name'):
            self.__name__ = script.get_global_name()
        else:
            raise ValueError("Invalid Script instance")

        if hasattr(script, '_get_language'):
            self.__language__ = script._get_language()
        elif hasattr(script, 'get_language'):
            self.__language__ = script.get_language()
        else:
            raise ValueError("Invalid Script instance")

        self._info = _Memory(cython.sizeof(GDExtensionScriptInstanceInfo3))

        cdef GDExtensionScriptInstanceInfo3 *info = <GDExtensionScriptInstanceInfo3 *>self._info.ptr
        cdef void *self_ptr = <void *><PyObject *>self

        self._base = NULL

        info.set_func = &ScriptInstance.set_callback
        info.get_func = &ScriptInstance.get_callback

        info.get_property_list_func = &ScriptInstance.get_property_list_callback
        info.property_can_revert_func = NULL
        info.property_get_revert_func = NULL
        info.get_property_state_func = NULL
        info.get_property_type_func = NULL
        info.validate_property_func = NULL

        info.free_property_list_func = NULL
        info.get_class_category_func = NULL

        info.get_owner_func = &ScriptInstance.get_owner_callback
        info.get_property_state_func = NULL

        info.get_method_list_func = &ScriptInstance.get_method_list_callback
        info.has_method_func = &ScriptInstance.has_method_callback
        info.get_method_argument_count_func = &ScriptInstance.get_method_argument_count_callback

        info.free_method_list_func = NULL

        info.call_func = &ScriptInstance.call_callback

        info.notification_func = &ScriptInstance.notification_callback

        info.to_string_func = &ScriptInstance.to_string_callback

        info.refcount_incremented_func = NULL
        info.refcount_decremented_func = NULL

        info.get_script_func = &ScriptInstance.get_script_callback

        info.is_placeholder_func = &ScriptInstance.is_placeholder_callback

        info.set_fallback_func = NULL
        info.get_fallback_func = NULL

        info.get_language_func = &ScriptInstance.get_language_callback

        info.free_func = &ScriptInstance.free_callback

        ref.Py_INCREF(self)
        self._base = gdextension_interface_script_instance_create3(info, self_ptr)

        print('INSTANCE CREATED')


    def __dealloc__(self):
        self.free()


    @staticmethod
    cdef uint8_t set_callback(void *p_instance, const void *p_name, const void *p_value) noexcept nogil:
        with gil:
            UtilityFunctions.print("set_callback")
            self = <object>p_instance
            name = type_funcs.string_name_to_pyobject(deref(<StringName *>p_name))
            value = type_funcs.variant_to_pyobject(deref(<Variant *>p_value))
            try:
                self.set(name, value)
            except Exception as exc:
                print_error_with_traceback(exc)
                return False

        return True

    @staticmethod
    cdef uint8_t get_callback(void *p_instance, const void *p_name, void *r_ret) noexcept nogil:
        with gil:
            UtilityFunctions.print("get_callback")
            self = <object>p_instance
            name = type_funcs.string_name_to_pyobject(deref(<StringName *>p_name))
            try:
                ret = self.get(name)
                type_funcs.variant_from_pyobject(ret, <Variant *>r_ret)
            except Exception as exc:
                print_error_with_traceback(exc)
                return False

            return True

    def set(self, name: AnyStr, value: Any) -> None:
        setattr(self.__owner__, name, value)

    def get(self, name: AnyStr) -> Any:
        return getattr(self.__owner__, name, self.__script_dict__.get(name))


    @staticmethod
    cdef const GDExtensionPropertyInfo *get_property_list_callback(void *p_instance, uint32_t *r_count) noexcept nogil:
        with gil:
            UtilityFunctions.print("get_property_list_callback")
            self = <object>p_instance
            try:
                property_list = self.get_property_list()
                (<ScriptInstance>self).property_info_data = _PropertyInfoDataArray(property_list)
                if r_count:
                    r_count[0] = <uint32_t>((<ScriptInstance>self).property_info_data).count
                return <const GDExtensionPropertyInfo *>((<ScriptInstance>self).property_info_data).ptr

            except Exception as exc:
                print_error_with_traceback(exc)

                if r_count:
                    r_count[0] = <uint32_t>0
                return NULL


    def get_property_list(self) -> List[PropertyInfo]:
        cdef list propinfo_list = []
        cdef VariantType vartype
        cdef uint32_t hint = 0, usage = 0

        for key, value in self.__script_dict__.items():
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
    cdef void *get_owner_callback(void *p_instance) noexcept nogil:
        with gil:
            UtilityFunctions.print("get_owner_callback")
            self = <object>p_instance
            return (<ScriptInstance>self).__owner__._owner

    @staticmethod
    cdef const GDExtensionMethodInfo *get_method_list_callback(void *p_instance, uint32_t *r_count) noexcept nogil:
        with gil:
            UtilityFunctions.print("method_list_callback")
            self = <object>p_instance
            try:
                method_list = self.get_method_list()
                (<ScriptInstance>self).method_info_data = _MethodInfoDataArray(method_list)
                if r_count:
                    r_count[0] = <uint32_t>((<ScriptInstance>self).method_info_data).count
                return <const GDExtensionMethodInfo *>((<ScriptInstance>self).method_info_data).ptr

            except Exception as exc:
                print_error_with_traceback(exc)

                if r_count:
                    r_count[0] = <uint32_t>0
                return NULL

    def method_list(self) -> List[MethodInfo]:
        cdef list methodinfo_list = []
        cdef VariantType argtype
        cdef int32_t id = 1

        for key, value in self.__script_dict__.items():
            if key.startswith('__') or not hasattr(value, '__code__'):
                continue

            annotations = getattr(value, '__annotations__', {})

            argnames = list(value.__code__.co_varnames)[1:]
            arguments = []

            for name in argnames:
                argtype = type_funcs.pytype_to_variant_type(annotations.get(name, None))
                arguments.append(PropertyInfo(argtype, name, self.__name__))

            methodinfo_list.append(MethodInfo(
                key,
                arguments,
                id,
                PropertyInfo(type_funcs.pytype_to_variant_type(annotations.get('return', None)))
            ))

            id += 1

        return methodinfo_list


    @staticmethod
    cdef uint8_t has_method_callback(void *p_instance, const void *p_name) noexcept nogil:
        with gil:
            UtilityFunctions.print("has_method_callback")
            self = <object>p_instance
            method_name = type_funcs.string_name_to_pyobject(deref(<StringName *>p_name))
            try:
                return self.has_method(method_name)
            except Exception as exc:
                print_error_with_traceback(exc)
                return False

    def has_method(self, method_name: AnyStr) -> bool:
        method = self.__script_dict__.get(method_name, None)

        return method is not None and hasattr(method, '__code__')

    @staticmethod
    cdef int64_t get_method_argument_count_callback(void *p_instance, const void *p_name, uint8_t *r_is_valid) noexcept nogil:
        with gil:
            UtilityFunctions.print("get_method_argument_count_callback")
            self = <object>p_instance
            method_name = type_funcs.string_name_to_pyobject(deref(<StringName *>p_name))
            try:
                ret = self.get_method_argument_count(method_name)
                r_is_valid[0] = ret > 0

                return ret
            except Exception as exc:
                print_error_with_traceback(exc)
                r_is_valid[0] = False
                return 0

    def get_method_argument_count(self, method_name: AnyStr) -> int:
        method = getattr(self.python_instance, method_name)
        if hasattr(method, '__code__'):
            arg_count  = method.__code__.co_argcount
            if arg_count > 1:
                return arg_count - 1

        raise ValueError("%r is not a valid ScriptInstance method" % method)


    @staticmethod
    cdef void call_callback(void *p_instance, const void *p_method, const (const void *) *p_args, int64_t p_count,
                            void *r_ret, GDExtensionCallError *r_error) noexcept nogil:
        with gil:
            self = <object>p_instance
            method_name = type_funcs.string_name_to_pyobject(deref(<StringName *>p_method))
            method = self.get_method(method_name)
            if method is not None:
                try:
                    bound_method = BoundPythonMethod(self.__owner__, method)
                except Exception as exc:
                    print_traceback_and_die(exc)

                _make_python_varcall(
                    <BoundPythonMethod>bound_method,
                    <const Variant **>p_args,
                    p_count,
                    <Variant *>r_ret,
                    r_error,
                    True
                )

    def get_method(self, method_name: AnyStr) -> Optional[CallablePythonType]:
        return self.__script_dict__.get(method_name, None)

    @staticmethod
    cdef void notification_callback(void *p_instance, int32_t p_what, uint8_t p_reversed) noexcept nogil:
        with gil:
            UtilityFunctions.print("notification_callback")
            self = <object>p_instance
            try:
                UtilityFunctions.print("notification_callback %r" % p_what)
                self.notification(p_what, p_reversed)
            except Exception as exc:
                print_error_with_traceback(exc)

    def notification(self, what: int, reversed: bool) -> None:
        raise NotImplementedError()

    @staticmethod
    cdef void to_string_callback(void *p_instance, uint8_t *r_is_valid, void *r_out) noexcept nogil:
        with gil:
            UtilityFunctions.print("to_string_callback")
            self = <object>p_instance
            try:
                type_funcs.string_from_pyobject(str(self), <String *>r_out)
                type_funcs.bool_from_pyobject(True, r_is_valid)
            except Exception as exc:
                print_error_with_traceback(exc)

    @staticmethod
    cdef void *get_script_callback(void *p_instance) noexcept nogil:
        with gil:
            UtilityFunctions.print("get_script_callback")
            self = <object>p_instance
            return (<ScriptInstance>self).__script__._owner

    @staticmethod
    cdef uint8_t is_placeholder_callback(void *p_instance) noexcept nogil:
        with gil:
            UtilityFunctions.print("is_placeholder_callback")
            self = <object>p_instance
            try:
                return self.is_placeholder()
            except Exception as exc:
                print_error_with_traceback(exc)

                return True

    def is_placeholder(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    cdef void *get_language_callback(void *p_instance) noexcept nogil:
        with gil:
            UtilityFunctions.print("get_language_callback")
            self = <object>p_instance
            return (<ScriptInstance>self).__language__._owner

    @staticmethod
    cdef void free_callback(void *p_instance) noexcept nogil:
        with gil:
            self = <object>p_instance
            self.free()

    def free(self):
        print('FREE')
        self._info.free()
        self.property_info_data.free()
        self.method_info_data.free()
        if self._base != NULL:
            ref.Py_DECREF(self)
            self._base = NULL


cdef object script_instance_to_pyobject(void *p_ptr):
    raise NotImplementedError()


cdef int script_instance_from_pyobject(ScriptInstance p_obj, void **r_ret) except -1:
    r_ret[0] = p_obj._base

    return 0
