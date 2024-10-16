cdef GDExtensionInstanceBindingCallbacks _Extension_binding_callbacks


cdef dict special_method_to_enum = {
    '_thread_enter': _THREAD_ENTER,
    '_thread_exit': _THREAD_EXIT,
    '_frame': _FRAME
}


cdef class Extension(Object):
    def __init__(self, ExtensionClass ext_class, Class base_class, bint notify=False, bint from_callback=False):
        if not isinstance(base_class, Class):
            raise TypeError("godot.Class instance is required for 'ext_class', got %r" % type(base_class))

        if not isinstance(ext_class, ExtensionClass):
            raise TypeError("ExtensionClass instance is required for 'ext_class', got %r" % type(ext_class))

        if not ext_class.is_registered:
            raise RuntimeError('Extension class must be registered')

        self._needs_cleanup = not from_callback

        self.__godot_class__ = ext_class

        cdef str class_name = ext_class.__name__
        cdef StringName _godot_class_name = StringName(class_name)

        cdef str base_class_name = base_class.__name__
        cdef StringName _godot_base_class_name = StringName(base_class_name)

        with nogil:
            self._owner = gdextension_interface_classdb_construct_object(_godot_base_class_name._native_ptr())

        if notify:
            notification = MethodBind(self, 'notification')
            notification(0, False) # NOTIFICATION_POSTINITIALIZE

        # INCREF because we lend a references of 'self' to Godot engine
        ref.Py_INCREF(self) # for set_instance, DECREF in ExtensionClass._free_instance

        cdef void *self_ptr = <void *><PyObject *>self

        with nogil:
            gdextension_interface_object_set_instance(self._owner, _godot_class_name._native_ptr(), self_ptr)

        _OBJECTDB[self.owner_id()] = self

        cdef object inner_init = self.__godot_class__.python_method_bindings.get('__inner_init__')

        if inner_init:
            inner_init(self)

        # print("%r initialized, from callback: %r" % (self, from_callback))


    cpdef destroy(self):
        with nogil:
            # Will call ExtensionClass._free_instance
            gdextension_interface_object_destroy(self._owner)
            self._owner = NULL
            self._needs_cleanup = False


    def __del__(self):
        if self._needs_cleanup and self._owner != NULL:
            print('Clean %r' % self)
            with nogil:
                # Will call ExtensionClass._free_instance
                gdextension_interface_object_destroy(self._owner)
                self._owner = NULL
                self._needs_cleanup = False

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
        cdef str name = p_name.py_str()

        cdef void* func_and_typeinfo_ptr

        # Special case, some virtual methods of ScriptLanguageExtension
        # which does not belong to Python ScriptLanguage implementations
        if name in special_method_to_enum:
            func_and_typeinfo_ptr = cls.get_special_method_info_ptr(special_method_to_enum[name])

            return func_and_typeinfo_ptr

        if name not in cls.virtual_method_implementation_bindings:
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
        Extension._call_virtual_with_data(p_instance, p_func, <const void **>p_args, r_ret)

    @staticmethod
    cdef void _call_virtual_with_data(void *p_self, void *p_func_and_info, const void **p_args,
                                      GDExtensionTypePtr r_ret) noexcept with gil:
        cdef tuple func_and_info = <tuple>p_func_and_info
        cdef object func = func_and_info[0]
        cdef SpecialMethod _special_func

        if isinstance(func, int):
            _special_func = <SpecialMethod>func
            with nogil:
                Extension._call_special_virtual(_special_func)
            return

        cdef tuple type_info = func_and_info[1]
        cdef object self = <object>p_self

        cdef size_t i = 0
        cdef list args = []

        cdef Variant variant_arg
        cdef GDExtensionBool bool_arg
        cdef int64_t int_arg
        cdef double float_arg
        cdef String string_arg
        cdef StringName stringname_arg

        cdef Dictionary dictionary_arg
        cdef Array array_arg
        cdef PackedStringArray packstringarray_arg
        cdef Object object_arg
        cdef Extension ext_arg
        cdef void *void_ptr_arg

        cdef size_t size = func.__code__.co_argcount - 1
        if size != (len(type_info) - 1):
            msg = (
                'Virtual method %r: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                    % (func, size, len(type_info) - 1, type_info[1:], type_info[0])
            )
            UtilityFunctions.printerr(msg)
            raise TypeError(msg)

        cdef str arg_type
        for i in range(size):
            arg_type = type_info[i + 1]
            if arg_type == 'float':
                float_arg = deref(<double *>p_args[i])
                args.append(float_arg)
            elif arg_type == 'String':
                string_arg = deref(<String *>p_args[i])
                args.append(string_arg.py_str())
            elif arg_type == 'StringName':
                stringname_arg = deref(<StringName *>p_args[i])
                args.append(stringname_arg.py_str())
            elif arg_type == 'bool':
                bool_arg = deref(<GDExtensionBool *>p_args[i])
                args.append(bool(bool_arg))
            elif arg_type == 'int' or arg_type == 'RID' or arg_type[6:] in _global_enum_info:
                int_arg = deref(<int64_t *>p_args[i])
                args.append(int_arg)
            elif arg_type in _global_inheritance_info:
                void_ptr_arg = deref(<void **>p_args[i])
                object_arg = _OBJECTDB.get(<uint64_t>void_ptr_arg, None)
                # print("Process %s argument %d in %r: %r" % (arg_type, i, func, object_arg))
                if object_arg is None and void_ptr_arg != NULL:
                    object_arg = Object(arg_type, from_ptr=<uint64_t>void_ptr_arg)
                    # print("Created %s argument from pointer %X: %r" % (arg_type, <uint64_t>void_ptr_arg, object_arg))
                args.append(object_arg)
            else:
                UtilityFunctions.push_error(
                    "NOT IMPLEMENTED: Can't convert %r arguments in virtual functions yet" % arg_type
                )
                args.append(None)

        cdef object result = func(self, *args)

        cdef str return_type = type_info[0]

        if return_type == 'PackedStringArray':
            packstringarray_arg = PackedStringArray(result)
            (<PackedStringArray *>r_ret)[0] = packstringarray_arg
        elif return_type == 'bool':
            bool_arg = bool(result)
            (<GDExtensionBool *>r_ret)[0] = bool_arg
        elif return_type == 'int' or return_type == 'RID' or return_type.startswith('enum:'):
            int_arg = result
            (<int64_t *>r_ret)[0] = int_arg
        elif return_type == 'String':
            string_arg = <String>result
            (<String *>r_ret)[0] = string_arg
        elif return_type == 'StringName':
            stringname_arg = <StringName>result
            (<StringName *>r_ret)[0] = stringname_arg
        elif return_type == 'Array' or return_type.startswith('typedarray:'):
            variant_arg = Variant(result)
            array_arg = <Array>variant_arg
            (<Array *>r_ret)[0] = array_arg
        elif return_type == 'Dictionary':
            variant_arg = Variant(result)
            dictionary_arg = <Dictionary>variant_arg
            (<Dictionary *>r_ret)[0] = dictionary_arg
        elif return_type == 'Variant':
            variant_arg = Variant(result)
            (<Variant *>r_ret)[0] = variant_arg
        elif return_type in _global_inheritance_info and isinstance(result, Object):
            object_arg = <Object>result
            (<void **>r_ret)[0] = object_arg._owner
        elif return_type in _global_inheritance_info and result is None:
            UtilityFunctions.push_warning("Expected %r but %r returned %r" % (return_type, func, result))

        elif return_type != 'Nil':
            if return_type in _global_inheritance_info:
                UtilityFunctions.push_error(
                    "NOT IMPLEMENTED: Can't convert %r from %r in %r" % (return_type, result, func)
                )

            else:
                UtilityFunctions.push_error(
                    "NOT IMPLEMENTED: "
                    ("Can't convert %r return types in virtual functions yet. Result was: %r in function %r"
                        % (return_type, result, func))
                )
