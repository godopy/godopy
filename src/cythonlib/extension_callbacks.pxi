cdef _gde_bool _ext_set_bind(void *p_instance, _gde_const_sn_ptr p_name, _gde_const_var_ptr p_value) noexcept nogil:
    if p_instance:
        # TODO: set instance property
        with gil:
            return _extgil_set_bind(p_instance, p_name, p_value)

cdef _gde_bool _extgil_set_bind(void *p_instance, _gde_const_sn_ptr p_name, _gde_const_var_ptr p_value) except -1:
    cdef object wrapper = <object>p_instance
    cdef str name = deref(<StringName *>p_name).py_str()
    cdef object value = deref(<Variant *>p_value).pythonize()
    print('SET BIND %r %s %r' % (wrapper, name, value))

    return False


cdef _gde_bool _ext_get_bind(void *p_instance, _gde_const_sn_ptr p_name, _gde_var_ptr r_ret) noexcept nogil:
    if p_instance:
        # TODO: get instance property
        with gil:
            return _extgil_get_bind(p_instance, p_name, r_ret)

cdef _gde_bool _extgil_get_bind(void *p_instance, _gde_const_sn_ptr p_name, _gde_var_ptr r_ret) except -1:
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


cdef GDExtensionBool _ext_property_can_revert_bind(void *p_instance, _gde_const_sn_ptr p_name) noexcept nogil:
    return False


cdef GDExtensionBool _ext_property_get_revert_bind(void *p_instance, _gde_const_sn_ptr p_name, _gde_var_ptr r_ret) noexcept nogil:
    return False


cdef GDExtensionBool _ext_validate_property_bind(void *p_instance, GDExtensionPropertyInfo *p_property) noexcept nogil:
    return False


cdef void _ext_notification_bind(void *p_instance, int32_t p_what, _gde_bool p_reversed) noexcept nogil:
    if p_instance:
        with gil:
            _extgil_notification_bind(p_instance, p_what, p_reversed)

cdef int _extgil_notification_bind(void *p_instance, int32_t p_what, _gde_bool p_reversed) except -1:
    cdef object wrapper = <object>p_instance
    # print("NOTIFICATION %r %d %s" % (<uint64_t>p_instance, p_what, p_reversed))

    cls = gd.Class('Object')
    cdef gd.MethodBind mb = gd.MethodBind(wrapper, 'notification', _methods_data=cls._methods)

    # if p_what == 0:
    #     print("NOTIFICATION POSTINITIALIZE")
    #     mb._call_internal_nil_int_bool(0, False) # NOTIFICATION_POSTINITIALIZE

    return 0


cdef void _ext_to_string_bind(void *p_instance, _gde_bool *r_is_valid, GDExtensionStringPtr r_out) noexcept nogil:
    if p_instance:
        with gil:
            _extgil_to_string_bind(p_instance, r_is_valid, r_out)

cdef int _extgil_to_string_bind(void *p_instance, _gde_bool *r_is_valid, GDExtensionStringPtr r_out) except -1:
    cdef object wrapper = <object>p_instance
    cdef str _repr = repr(wrapper)
    print("TO_STRING %r %x" % (wrapper, <uint64_t>p_instance))
    cdef GDExtensionBool is_valid = deref(r_is_valid)
    cdef String out = deref(<String *>r_out)
    is_valid = True
    out = String(_repr)

    return 0


cdef void *_ext_get_virtual_call_data(void *p_userdata, _gde_const_sn_ptr p_name) noexcept nogil:
    if p_name == NULL:
        return NULL
    cdef StringName name = deref(<StringName *>p_name)
    with gil:
        return _extgil_get_virtual_call_data(p_userdata, name.py_str())

cdef void *_extgil_get_virtual_call_data(void *p_cls, str name) except NULL:
    cdef ExtensionClass cls = <ExtensionClass>p_cls
    for fname, func in cls.virtual_method_implementation_bindings.iteritems():
        if fname == name:
            ref.Py_INCREF(func)
            return <void *><PyObject *>func

cdef void _ext_call_virtual_with_data(
    void *p_instance,
    _gde_const_sn_ptr p_name,
    void *p_func,
    GDExtensionConstTypePtr *p_args,
    GDExtensionTypePtr r_ret
) noexcept nogil:
    if p_name == NULL:
        return
    cdef StringName name = deref(<StringName *>p_name)
    with gil:
        _extgil_call_virtual_with_data(p_instance, name.py_str(), p_func, <const void **>p_args, r_ret)

cdef int _extgil_call_virtual_with_data(
    void *p_instance,
    str p_name,
    void *p_func,
    const void **p_args,
    GDExtensionTypePtr r_ret
) except -1:
    cdef object wrapper = <object>p_instance
    # gd._printerr('call %s' % wrapper)
    cdef func = <object>p_func
    cdef size_t i = 0
    cdef list args = []
    cdef real_t arg
    cdef size_t size = func.__code__.co_argcount - 1
    if size < 0:
        gd._printerr('Wrong number of arguments %d' % size)
        raise TypeError('Wrong number of arguments %d' % size)
        return -1

    for i in range(size):
        arg = deref(<real_t *>p_args[i])
        args.append(<object><Variant>arg)
        i += 1

    cdef object res = func(wrapper._wrapped, *args)
    # cdef Variant ret = <Variant>res

    # ref.Py_DECREF(func)

    #TODO: Real type, not variant
    # cdef Variant *ptr = deref(<Variant **>r_ret)
    # ptr = &ret

    return 0
