cdef GDExtensionBool _ext_set_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value) noexcept nogil:
    if p_instance:
        # TODO: set instance property
        with gil:
            return _extgil_set_bind(p_instance, p_name, p_value)

cdef GDExtensionBool _extgil_set_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value) except -1:
    cdef object wrapper = <object>p_instance
    cdef str name = deref(<StringName *>p_name).py_str()
    cdef object value = deref(<Variant *>p_value).pythonize()
    print('SET BIND %r %s %r' % (wrapper, name, value))

    return False


cdef GDExtensionBool _ext_get_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) noexcept nogil:
    if p_instance:
        # TODO: get instance property
        with gil:
            return _extgil_get_bind(p_instance, p_name, r_ret)

cdef GDExtensionBool _extgil_get_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) except -1:
    cdef object wrapper = <object>p_instance
    cdef str name = deref(<StringName *>p_name).py_str()
    print('GET BIND %r %s' % (wrapper, name))

    return False


cdef GDExtensionPropertyInfo *_ext_get_property_list_bind(void *p_instance, uint32_t *r_count) noexcept nogil:
    if r_count == NULL:
        return NULL

    cdef uint32_t count = deref(r_count)
    with gil:
        print('GETPROPLIST %x %d' % (<uint64_t>p_instance))
    if not p_instance:
        count = 0
        return NULL
    # TODO: Create and return property list
    count = 0
    return NULL


cdef void _ext_free_property_list_bind(void *p_instance, const GDExtensionPropertyInfo *p_list, uint32_t p_count) noexcept nogil:
    if p_instance:
        with gil:
            print('FREEPROPLIST %x' % (<uint64_t>p_instance))


cdef GDExtensionBool _ext_property_can_revert_bind(void *p_instance, GDExtensionConstStringNamePtr p_name) noexcept nogil:
    return False


cdef GDExtensionBool _ext_property_get_revert_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) noexcept nogil:
    return False


cdef GDExtensionBool _ext_validate_property_bind(void *p_instance, GDExtensionPropertyInfo *p_property) noexcept nogil:
    return False


cdef void _ext_notification_bind(void *p_instance, int32_t p_what, GDExtensionBool p_reversed) noexcept nogil:
    if p_instance:
        with gil:
            _extgil_notification_bind(p_instance, p_what, p_reversed)

cdef int _extgil_notification_bind(void *p_instance, int32_t p_what, GDExtensionBool p_reversed) except -1:
    cdef object wrapper = <object>p_instance
    # print("NOTIFICATION %r %d %s" % (<uint64_t>p_instance, p_what, p_reversed))

    return 0


cdef void _ext_to_string_bind(void *p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) noexcept nogil:
    if p_instance:
        with gil:
            _extgil_to_string_bind(p_instance, r_is_valid, r_out)

cdef int _extgil_to_string_bind(void *p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) except -1:
    cdef object wrapper = <object>p_instance
    cdef str _repr = repr(wrapper)
    print("TO_STRING %r %x" % (wrapper, <uint64_t>p_instance))
    cdef GDExtensionBool is_valid = deref(r_is_valid)
    cdef String out = deref(<String *>r_out)
    is_valid = True
    out = String(_repr)

    return 0


cdef void *_ext_get_virtual_call_data(void *p_userdata, GDExtensionConstStringNamePtr p_name) noexcept nogil:
    cdef StringName name = deref(<StringName *>p_name)

    # Create PyThreadState for every Godot thread,
    # otherwise calling GIL function from different threads would create a deadlock
    PythonRuntime.get_singleton().ensure_current_thread_state()

    return _extgil_get_virtual_call_data(p_userdata, name)


cdef void *_extgil_get_virtual_call_data(void *p_cls, const StringName &p_name) noexcept with gil:
    cdef ExtensionClass cls = <ExtensionClass>p_cls
    cdef str name = p_name.py_str()
    cdef object func = cls.virtual_method_implementation_bindings.get(name)
    if func is None:
        return NULL

    cdef dict func_info = cls.__inherits__.get_method_info(name)
    cdef tuple func_and_info = (func, func_info['type_info'])
    ref.Py_INCREF(func_and_info)
    # TODO: Store the pointer and decref when freeing the instance

    return <void *><PyObject *>func_and_info


cdef void _ext_call_virtual_with_data(void *p_instance, GDExtensionConstStringNamePtr p_name, void *p_func, GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) noexcept nogil:
    _extgil_call_virtual_with_data(p_instance, p_func, <const void **>p_args, r_ret)

cdef void _extgil_call_virtual_with_data(void *p_instance, void *p_func_and_info, const void **p_args, GDExtensionTypePtr r_ret) noexcept with gil:
    cdef object wrapper = <object>p_instance
    cdef func_and_info = <tuple>p_func_and_info
    cdef object func = func_and_info[0]
    cdef tuple type_info = func_and_info[1]
    cdef size_t i = 0
    cdef list args = []

    cdef Variant variant_arg
    cdef double float_arg
    cdef String string_arg
    cdef StringName stringname_arg
    cdef GDExtensionBool bool_arg
    cdef int64_t int_arg
    cdef PackedStringArray packstringarray_arg

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

    cdef object res = func(wrapper, *args)

    cdef str return_type = type_info[0]

    if return_type == 'PackedStringArray':
        packstringarray_arg = PackedStringArray(res)
        (<PackedStringArray *>r_ret)[0] = packstringarray_arg
    elif return_type == 'bool':
        bool_arg = bool(res)
        (<GDExtensionBool *>r_ret)[0] = bool_arg
    elif return_type == 'String':
        string_arg = <String>res
        (<String *>r_ret)[0] = string_arg
    elif return_type == 'Variant' or return_type == 'Object':
        # ResourceFormatLoader._load expects Variant, but Object is declared
        variant_arg = Variant(res)
        (<Variant *>r_ret)[0] = variant_arg

    elif return_type != 'Nil':
        UtilityFunctions.printerr("NOT IMPLEMENTED: Can't convert %r return types in virtual functions yet" % return_type)
