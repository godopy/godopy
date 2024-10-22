ctypedef fused gdcallable_ft:
    MethodBind
    UtilityFunction
    BuiltinMethod


ctypedef void (*_ptrcall_func)(gdcallable_ft, void *, const void **, size_t) noexcept nogil
ctypedef void (*_varcall_func)(gdcallable_ft, const Variant **, size_t, Variant *, GDExtensionCallError *) noexcept nogil


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

    cdef Variant *vargs = <Variant *>gdextension_interface_mem_alloc(size * cython.sizeof(Variant))

    for i in range(size):
        vargs[i] = Variant(<const PyObject *>args[i])

    varcall(method, <const Variant **>&vargs, size, &ret, &err)

    gdextension_interface_mem_free(vargs)

    if err.error != GDEXTENSION_CALL_OK:
        raise RuntimeError(ret.pythonize())

    return ret.pythonize()


cdef inline object _make_engine_ptrcall(gdcallable_ft method, _ptrcall_func ptrcall, tuple args):
    """
    Implements GDExtension's 'ptrcall' logic when calling Engine methods from Python
    """
    cdef tuple type_info = method.type_info
    cdef size_t i = 0, size = len(args)

    # UtilityFunctions.print("Variant: %d, String: %d" % (cython.sizeof(Variant), cython.sizeof(String)))

    if (size != len(method.type_info) - 1):
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__,  size, len(type_info) - 1,
                    type_info[1:], type_info[0])
        )
        UtilityFunctions.printerr(msg)
        raise TypeError(msg)

    cdef GDExtensionUninitializedTypePtr *ptr_args = <GDExtensionUninitializedTypePtr *> \
        gdextension_interface_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

    if ptr_args == NULL:
        raise MemoryError("Not enough memory")

    cdef str return_type = 'Nil'
    cdef str arg_type = 'Nil'
    cdef bint unknown_argtype_error = False
    cdef bint unknown_type_error = False

    cdef int arg_typecode = 0

    cdef uint8_t bool_arg
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

    # cdef Object object_arg
    # cdef Extension ext_arg
    cdef void *ptr_arg
    cdef Variant vaiant_arg
    cdef object pyarg

    # Optimized get_node for Python nodes
    if method.__name__ == 'get_node' and size == 1 and args[0] in _NODEDB:
        pyarg = _NODEDB[args[0]]
        return pyarg

    # TODO: Optimize
    for i in range(size):
        arg_type = type_info[i + 1]
        pyarg = args[i]

        if arg_type == 'bool':
            type_funcs.bool_from_pyobject(pyarg, &bool_arg)
            ptr_args[i] = &bool_arg
        elif arg_type == 'int' or arg_type == 'RID' or arg_type.startswith('enum:'):
            type_funcs.int_from_pyobject(pyarg, &int_arg)
            ptr_args[i] = &int_arg
        elif arg_type == 'float':
            type_funcs.float_from_pyobject(pyarg, &float_arg)
            ptr_args[i] = &float_arg
        elif arg_type == 'String':
            type_funcs.string_from_pyobject(pyarg, &string_arg)
            ptr_args[i] = &string_arg
        elif arg_type == 'Vector2':
            type_funcs.vector2_from_pyobject(pyarg, &vector2_arg)
            ptr_args[i] = &vector2_arg
        elif arg_type == 'Vector2i':
            type_funcs.vector2i_from_pyobject(pyarg, &vector2i_arg)
            ptr_args[i] = &vector2i_arg
        elif arg_type == 'Rect2':
            type_funcs.rect2_from_pyobject(pyarg, &rect2_arg)
            ptr_args[i] = &rect2_arg
        elif arg_type == 'Rect2i':
            type_funcs.rect2i_from_pyobject(pyarg, &rect2i_arg)
            ptr_args[i] = &rect2i_arg
        elif arg_type == 'StringName':
            stringname_arg = <StringName>pyarg
            ptr_args[i] = &stringname_arg
        elif arg_type == 'NodePath':
            nodepath_arg = <NodePath>pyarg
            ptr_args[i] = &nodepath_arg
        elif arg_type == 'Variant':
            variant_arg = Variant(<const PyObject *>pyarg)
            ptr_args[i] = &variant_arg
        elif arg_type in _global_inheritance_info:
            object_from_pyobject(pyarg, &ptr_arg)
            ptr_args[i] = &ptr_arg

        else:
            unknown_argtype_error = True
            break

    if unknown_argtype_error:
        gdextension_interface_mem_free(ptr_args)
        UtilityFunctions.printerr(
            "Don't know how to convert %r types, passed arg was: %r" % (arg_type, pyarg)
        )
        raise NotImplementedError("Don't know how to convert %r types" % arg_type)

    return_type = method.type_info[0]

    cdef bint return_variant = False
    if return_type == 'Nil':
        ptrcall(method, NULL, <const void **>ptr_args, size)
    elif return_type == 'Variant':
        ptrcall(method, &variant_arg, <const void **>ptr_args, size)
        return_variant = True
    elif return_type == 'bool':
        ptrcall(method, &bool_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.bool_to_pyobject(bool_arg)
    elif return_type == 'int' or return_type == 'RID' or return_type[6:] in _global_enum_info:
        ptrcall(method, &int_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.int_to_pyobject(int_arg)
    elif return_type == 'float':
        ptrcall(method, &float_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.float_to_pyobject(float_arg)
    elif return_type == 'String':
        ptrcall(method, &string_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.string_to_pyobject(string_arg)
    elif return_type == 'Vector2':
        ptrcall(method, &vector2_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector2_to_pyobject(vector2_arg)
    elif return_type == 'Vector2i':
        ptrcall(method, &vector2i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector2i_to_pyobject(vector2i_arg)
    elif return_type == 'Rect2':
        ptrcall(method, &rect2_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rect2_to_pyobject(rect2_arg)
    elif return_type == 'Rect2i':
        ptrcall(method, &rect2i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rect2i_to_pyobject(rect2i_arg)
    elif return_type == 'PackedStringArray':
        ptrcall(method, &packed_string_array_arg, <const void **>ptr_args, size)
        variant_arg = Variant(packed_string_array_arg)
        return_variant = True
    elif return_type in _global_inheritance_info:
        ptrcall(method, &ptr_arg, <const void **>ptr_args, size)
        pyarg = object_to_pyobject(ptr_arg)
    else:
        unknown_type_error = True

    gdextension_interface_mem_free(ptr_args)

    if unknown_type_error:
        UtilityFunctions.printerr("Don't know how to return %r types. Returning None." % return_type)
        # raise NotImplementedError("Don't know how to return %r types" % return_type)
        return

    if return_type == 'Nil':
        return None
    elif return_variant:
        return variant_arg.pythonize()

    return pyarg
