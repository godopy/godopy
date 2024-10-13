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
        self._godot_class_name = StringName(class_name)

        cdef str base_class_name = base_class.__name__
        self._godot_base_class_name = StringName(base_class_name)

        with nogil:
            self._owner = gdextension_interface_classdb_construct_object(self._godot_base_class_name._native_ptr())

        if notify:
            notification = MethodBind(self, 'notification')
            notification(0, False) # NOTIFICATION_POSTINITIALIZE

        ref.Py_INCREF(self) # DECREF in ExtensionClass._free_instance

        cdef void *self_ptr = <void *><PyObject *>self

        with nogil:
            gdextension_interface_object_set_instance(self._owner, self._godot_class_name._native_ptr(), self_ptr)

        cdef object impl_init_0 = self.__godot_class__.python_method_bindings.get('__init__')
        cdef object imple_init_1 = self.__godot_class__.python_method_bindings.get('_init')
        if impl_init_0:
            impl_init_0(self)
        if imple_init_1:
            imple_init_1(self)

        print("%r initialized, from callback: %r" % (self, from_callback))


    cpdef destroy(self):
        with nogil:
            # Will call ExtensionClass._free_instance
            gdextension_interface_object_destroy(self._owner)
            self._owner = NULL
            self._needs_cleanup = False


    def __del__(self):
        if self._needs_cleanup:
            print('Clean %r' % self)
            with nogil:
                # Will call ExtensionClass._free_instance
                gdextension_interface_object_destroy(self._owner)
                self._owner = NULL
                self._needs_cleanup = False

    @staticmethod
    cdef void *get_virtual_call_data(void *p_userdata, GDExtensionConstStringNamePtr p_name) noexcept nogil:
        cdef StringName name = deref(<StringName *>p_name)

        # Create PyThreadState for every Godot thread,
        # otherwise calling GIL function from different threads would create a deadlock
        PythonRuntime.get_singleton().ensure_current_thread_state()

        return Extension._get_virtual_call_data(p_userdata, name)

    @staticmethod
    cdef void *_get_virtual_call_data(void *p_cls, const StringName &p_name) noexcept with gil:
        cdef ExtensionClass cls = <ExtensionClass>p_cls
        cdef str name = p_name.py_str()
        if name not in cls.virtual_method_implementation_bindings:
            return NULL

        cdef void* func_and_typeinfo_ptr = cls.get_method_and_method_type_info_ptr(name)

        return func_and_typeinfo_ptr

    @staticmethod
    cdef void call_virtual_with_data(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name,
                                     void *p_func, const GDExtensionConstTypePtr *p_args,
                                     GDExtensionTypePtr r_ret) noexcept nogil:
        Extension._call_virtual_with_data(p_instance, p_func, <const void **>p_args, r_ret)

    @staticmethod
    cdef void _call_virtual_with_data(void *p_self, void *p_func_and_info, const void **p_args,
                                      GDExtensionTypePtr r_ret) noexcept with gil:
        cdef object self = <object>p_self
        cdef tuple func_and_info = <tuple>p_func_and_info
        cdef object func = func_and_info[0]
        cdef tuple type_info = func_and_info[1]

        cdef size_t i = 0
        cdef list args = []

        cdef Variant variant_arg
        cdef GDExtensionBool bool_arg
        cdef int64_t int_arg
        cdef double float_arg
        cdef String string_arg
        cdef StringName stringname_arg

        cdef PackedStringArray packstringarray_arg
        cdef Extension ext_arg

        cdef size_t size = func.__code__.co_argcount - 1
        if size < 0 or size != (len(type_info) - 1):
            UtilityFunctions.printerr('Wrong number of arguments %d' % size)
            raise TypeError('Wrong number of arguments %d' % size)

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
            elif arg_type == 'int':
                int_arg = deref(<int64_t *>p_args[i])
                args.append(int_arg)
            else:
                UtilityFunctions.printerr("NOT IMPLEMENTED: Can't convert %r arguments in virtual functions yet" % arg_type)
                args.append(None)

        cdef object res = func(self, *args)

        cdef str return_type = type_info[0]

        if return_type == 'PackedStringArray':
            packstringarray_arg = PackedStringArray(res)
            (<PackedStringArray *>r_ret)[0] = packstringarray_arg
        elif return_type == 'bool':
            bool_arg = bool(res)
            (<GDExtensionBool *>r_ret)[0] = bool_arg
        elif return_type == 'int' or return_type == 'RID':
            int_arg = res
            (<int64_t *>r_ret)[0] = int_arg
        elif return_type == 'String':
            string_arg = <String>res
            (<String *>r_ret)[0] = string_arg
        elif return_type == 'Variant' or return_type == 'Object':
            # FIXME: ResourceFormatLoader._load expects Variant, but Object is declared
            variant_arg = Variant(res)
            (<Variant *>r_ret)[0] = variant_arg

        elif return_type != 'Nil':
            UtilityFunctions.printerr("NOT IMPLEMENTED: Can't convert %r return types in virtual functions yet" % return_type)
