ctypedef fused gdcallable_ft:
    MethodBind
    UtilityFunction
    BuiltinMethod


ctypedef void (*_ptrcall_func)(gdcallable_ft, GDExtensionTypePtr, GDExtensionConstTypePtr *, size_t) noexcept nogil
ctypedef void (*_varcall_func)(gdcallable_ft, const GDExtensionConstVariantPtr *, size_t,
                               GDExtensionUninitializedVariantPtr, GDExtensionCallError *) noexcept nogil


cdef class EngineCallableBase:
    """
    Base class for MethodBind, UtilityFunction and BuiltinMethod.
    """
    def __init__(self):
        raise NotImplementedError("Base class, cannot instantiate")


cdef inline object _make_engine_varcall(gdcallable_ft method, _varcall_func varcall, tuple args):
    """
    Implements GDExtension's 'call' logic when calling Engine methods from Python
    """
    cdef Variant ret
    cdef GDExtensionCallError err

    err.error = GDEXTENSION_CALL_OK

    cdef size_t i = 0, size = len(args)

    cdef Variant *_args = <Variant *>gdextension_interface_mem_alloc(size * cython.sizeof(Variant))

    for i in range(size):
        _args[i] = Variant(<const PyObject *>args[i])

    varcall(method, <const GDExtensionConstVariantPtr *>&_args, size, &ret, &err)

    gdextension_interface_mem_free(_args)

    if err.error != GDEXTENSION_CALL_OK:
        raise RuntimeError(ret.pythonize())

    return ret.pythonize()


cdef inline object _make_engine_ptrcall(gdcallable_ft method, _ptrcall_func ptrcall, tuple args):
    """
    Implements GDExtension's 'ptrcall' logic when calling Engine methods from Python
    """
    cdef tuple type_info = method.type_info
    cdef Variant arg
    cdef size_t i = 0, size = len(args)

    if (size != len(method.type_info) - 1):
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__,  size, len(type_info) - 1,
                    type_info[1:], type_info[0])
        )
        UtilityFunctions.printerr(msg)
        raise TypeError(msg)

    cdef GDExtensionUninitializedTypePtr *p_args = <GDExtensionUninitializedTypePtr *> \
        gdextension_interface_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

    cdef str return_type = 'Nil'
    cdef str arg_type = 'Nil'
    cdef bint unknown_argtype_error = False
    cdef bint unknown_type_error = False

    cdef int arg_typecode = 0

    cdef GDExtensionBool bool_arg
    cdef int64_t int_arg
    cdef double float_arg
    cdef String string_arg
    cdef Vector2 vector2_arg
    cdef Vector2i vector2i_arg
    cdef Rect2 rect2_arg
    cdef Rect2i rect2i_arg

    cdef StringName stringname_arg
    cdef NodePath nodepath_arg

    cdef PackedStringArray packed_string_array_arg

    cdef Object object_arg
    cdef Extension ext_arg
    cdef void *void_ptr_arg

    cdef double x, y, z, w
    cdef int32_t xi, yi, zi, wi

    cdef object pyarg

    # Optimized get_node for Python nodes
    if method.__name__ == 'get_node' and size == 1 and args[0] in _NODEDB:
        pyarg = _NODEDB[args[0]]
        # print("'get_node' shortcut for %r" % pyarg)
        return pyarg

    # TODO: Optimize
    for i in range(size):
        arg_type = type_info[i + 1]
        pyarg = args[i]

        if arg_type == 'bool':
            bool_arg = arg.booleanize()
            p_args[i] = &bool_arg
        elif arg_type == 'int' or arg_type == 'RID' or arg_type.startswith('enum:'):
            int_arg = pyarg
            p_args[i] = &int_arg
        elif arg_type == 'float':
            float_arg = pyarg
            p_args[i] = &float_arg
        elif arg_type == 'String':
            string_arg = <String>pyarg
            p_args[i] = &string_arg
        elif arg_type == 'Vector2':
            x, y = pyarg
            vector2_arg = Vector2(x, y)
            p_args[i] = &vector2_arg
        elif arg_type == 'Vector2i':
            xi, yi = pyarg
            vector2i_arg = Vector2i(xi, yi)
            p_args[i] = &vector2i_arg
        elif arg_type == 'Rect2':
            position, size = pyarg
            x, y = position
            z, w = size
            rect2_arg = Rect2(x, y, z, w)
            p_args[i] = &rect2_arg
        elif arg_type == 'Rect2i':
            position, size = pyarg
            xi, yi = position
            zi, wi = size
            rect2i_arg = Rect2i(xi, yi, zi, wi)
            p_args[i] = &rect2_arg
        elif arg_type == 'StringName':
            stringname_arg = <StringName>pyarg
            p_args[i] = &stringname_arg
        elif arg_type == 'NodePath':
            nodepath_arg = <NodePath>pyarg
            p_args[i] = &nodepath_arg
        elif arg_type == 'Variant':
            arg = Variant(<const PyObject *>pyarg)
            p_args[i] = &arg
        elif arg_type in _global_inheritance_info and isinstance(pyarg, Object):
            object_arg = <Object>pyarg
            p_args[i] = &object_arg._owner

        else:
            unknown_argtype_error = True
            break

    if unknown_argtype_error:
        gdextension_interface_mem_free(p_args)
        UtilityFunctions.printerr(
            "Don't know how to convert %r types, passed arg was: %r" % (arg_type, pyarg)
        )
        raise NotImplementedError("Don't know how to convert %r types" % arg_type)

    return_type = method.type_info[0]

    if return_type == 'Nil':
        ptrcall(method, NULL, <GDExtensionConstTypePtr *>p_args, size)
    elif return_type == 'Variant':
        ptrcall(method, &arg, <GDExtensionConstTypePtr *>p_args, size)
    elif return_type == 'String':
        ptrcall(method, &string_arg, <GDExtensionConstTypePtr *>p_args, size)
        arg = <Variant>string_arg
    elif return_type == 'float':
        ptrcall(method, &float_arg, <GDExtensionConstTypePtr *>p_args, size)
        arg = <Variant>float_arg
    elif return_type == 'int' or return_type == 'RID' or return_type[6:] in _global_enum_info:
        ptrcall(method, &int_arg, <GDExtensionConstTypePtr *>p_args, size)
        arg = <Variant>int_arg
    elif return_type == 'bool':
        ptrcall(method, &bool_arg, <GDExtensionConstTypePtr *>p_args, size)
        arg = <Variant>bool_arg
    elif return_type == 'Vector2':
        ptrcall(method, &vector2_arg, <GDExtensionConstTypePtr *>p_args, size)
        arg = <Variant>vector2_arg
    elif return_type == 'PackedStringArray':
        ptrcall(method, &packed_string_array_arg, <GDExtensionConstTypePtr *>p_args, size)
        arg = <Variant>packed_string_array_arg
    elif return_type in _global_inheritance_info:
        # print("Calling from %r with %r, receiving %s" % (gdcallable, args, return_type))
        ptrcall(method, &void_ptr_arg, <GDExtensionConstTypePtr *>p_args, size)
        object_arg = _OBJECTDB.get(<uint64_t>void_ptr_arg, None)
        # print("Process %s return value %r" % (return_type, object_arg))
        if object_arg is None and void_ptr_arg != NULL:
            object_arg = Object(return_type, from_ptr=<uint64_t>void_ptr_arg)
            # print("Created %s return value from pointer %X: %r" % (return_type, <uint64_t>void_ptr_arg, object_arg))
        gdextension_interface_mem_free(p_args)
        return object_arg
    else:
        unknown_type_error = True

    gdextension_interface_mem_free(p_args)

    if unknown_type_error:
        UtilityFunctions.printerr("Don't know how to return %r types. Returning None." % return_type)
        # raise NotImplementedError("Don't know how to return %r types" % return_type)
        return

    if return_type == 'Nil':
        return None

    return arg.pythonize()
