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
        type_funcs.variant_from_pyobject(args[i], &vargs[i])

    varcall(method, <const Variant **>&vargs, size, &ret, &err)

    gdextension_interface_mem_free(vargs)

    if err.error != GDEXTENSION_CALL_OK:
        raise RuntimeError(ret.pythonize())

    return ret.pythonize()


cdef inline object _make_engine_ptrcall(gdcallable_ft method, _ptrcall_func ptrcall, tuple args):
    """
    Implements GDExtension's 'ptrcall' logic when calling Engine methods from Python
    """
    cdef int8_t *type_info = method._type_info_opt
    cdef size_t i = 0, size = len(args)

    # UtilityFunctions.print("Variant: %d, String: %d" % (cython.sizeof(Variant), cython.sizeof(String)))

    if (size != len(method.type_info) - 1):
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__,  size, len(method.type_info) - 1,
                    method.type_info[1:], method.type_info[0])
        )
        UtilityFunctions.printerr(msg)
        raise TypeError(msg)

    cdef GDExtensionUninitializedTypePtr *ptr_args = <GDExtensionUninitializedTypePtr *> \
        gdextension_interface_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

    if ptr_args == NULL:
        raise MemoryError("Not enough memory")

    cdef bint unknown_argtype_error = False
    cdef bint unknown_type_error = False

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
    cdef _RID rid_arg
    cdef void *ptr_arg
    cdef Dictionary dictionary_arg
    cdef Array array_arg

    cdef PackedStringArray packed_string_array_arg

    cdef Variant variant_arg
    cdef object pyarg

    # Optimized get_node for Python nodes
    if method.__name__ == 'get_node' and size == 1 and args[0] in _NODEDB:
        pyarg = _NODEDB[args[0]]
        return pyarg

    cdef int8_t arg_type
    for i in range(size):
        arg_type = type_info[i + 1]
        pyarg = args[i]

        # NOTE: Cython compiles this to C switch/case
        if arg_type == BOOL:
            type_funcs.bool_from_pyobject(pyarg, &bool_arg)
            ptr_args[i] = &bool_arg
        elif arg_type == INT:
            type_funcs.int_from_pyobject(pyarg, &int_arg)
            ptr_args[i] = &int_arg
        elif arg_type == FLOAT:
            type_funcs.float_from_pyobject(pyarg, &float_arg)
            ptr_args[i] = &float_arg
        elif arg_type == STRING:
            type_funcs.string_from_pyobject(pyarg, &string_arg)
            ptr_args[i] = &string_arg
        elif arg_type == VECTOR2:
            type_funcs.vector2_from_pyobject(pyarg, &vector2_arg)
            ptr_args[i] = &vector2_arg
        elif arg_type == VECTOR2I:
            type_funcs.vector2i_from_pyobject(pyarg, &vector2i_arg)
            ptr_args[i] = &vector2i_arg
        elif arg_type == RECT2:
            type_funcs.rect2_from_pyobject(pyarg, &rect2_arg)
            ptr_args[i] = &rect2_arg
        elif arg_type == RECT2I:
            type_funcs.rect2i_from_pyobject(pyarg, &rect2i_arg)
            ptr_args[i] = &rect2i_arg
        elif arg_type == STRING_NAME:
            type_funcs.string_name_from_pyobject(pyarg, &stringname_arg)
            ptr_args[i] = &stringname_arg
        elif arg_type == NODE_PATH:
            type_funcs.node_path_from_pyobject(pyarg, &nodepath_arg)
            ptr_args[i] = &nodepath_arg
        elif arg_type == RID:
            type_funcs.rid_from_pyobject(pyarg, &rid_arg)
            ptr_args[i] = &rid_arg
        elif arg_type == OBJECT:
            object_from_pyobject(pyarg, &ptr_arg)
            ptr_args[i] = &ptr_arg
        elif arg_type == DICTIONARY:
            type_funcs.dictionary_from_pyobject(pyarg, &dictionary_arg)
            ptr_args[i] = &dictionary_arg
        elif arg_type == ARRAY:
            type_funcs.array_from_pyobject(pyarg, &array_arg)
            ptr_args[i] = &array_arg
        elif arg_type == PACKED_STRING_ARRAY:
            type_funcs.packed_string_array_from_pyobject(pyarg, &packed_string_array_arg)
            ptr_args[i] = &packed_string_array_arg
        elif arg_type == VARIANT_MAX:
            type_funcs.variant_from_pyobject(pyarg, &variant_arg)
            ptr_args[i] = &variant_arg
        else:
            unknown_argtype_error = True
            break

    if unknown_argtype_error:
        gdextension_interface_mem_free(ptr_args)
        UtilityFunctions.printerr(
            "NOT IMPLEMENTED: Don't know how to convert %r types, passed arg was: %r" % (arg_type, pyarg)
        )
        raise NotImplementedError("Don't know how to convert %r types" % arg_type)

    cdef int8_t return_type = type_info[0]

    # NOTE: Cython compiles this to C switch/case
    if return_type == NIL:
        ptrcall(method, NULL, <const void **>ptr_args, size)
    elif return_type == BOOL:
        ptrcall(method, &bool_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.bool_to_pyobject(bool_arg)
    elif return_type == INT:
        ptrcall(method, &int_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.int_to_pyobject(int_arg)
    elif return_type == FLOAT:
        ptrcall(method, &float_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.float_to_pyobject(float_arg)
    elif return_type == STRING:
        ptrcall(method, &string_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.string_to_pyobject(string_arg)
    elif return_type == VECTOR2:
        ptrcall(method, &vector2_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector2_to_pyobject(vector2_arg)
    elif return_type == VECTOR2I:
        ptrcall(method, &vector2i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector2i_to_pyobject(vector2i_arg)
    elif return_type == RECT2:
        ptrcall(method, &rect2_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rect2_to_pyobject(rect2_arg)
    elif return_type == RECT2I:
        ptrcall(method, &rect2i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rect2i_to_pyobject(rect2i_arg)
    elif return_type == STRING_NAME:
        ptrcall(method, &stringname_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.string_name_to_pyobject(stringname_arg)
    elif return_type == NODE_PATH:
        ptrcall(method, &nodepath_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.node_path_to_pyobject(nodepath_arg)
    elif return_type == RID:
        ptrcall(method, &rid_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rid_to_pyobject(rid_arg)
    elif return_type == OBJECT:
        ptrcall(method, &ptr_arg, <const void **>ptr_args, size)
        pyarg = object_to_pyobject(ptr_arg)
    elif return_type == DICTIONARY:
        ptrcall(method, &dictionary_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.dictionary_to_pyobject(dictionary_arg)
    elif return_type == ARRAY:
        ptrcall(method, &array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.array_to_pyobject(array_arg) 
    elif return_type == PACKED_STRING_ARRAY:
        ptrcall(method, &packed_string_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_string_array_to_pyobject(packed_string_array_arg)
    elif return_type == VARIANT_MAX:
        ptrcall(method, &variant_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.variant_to_pyobject(variant_arg)
    else:
        unknown_type_error = True

    gdextension_interface_mem_free(ptr_args)

    if unknown_type_error:
        UtilityFunctions.printerr("Don't know how to return %r types. Returning None." % return_type)
        # raise NotImplementedError("Don't know how to return %r types" % return_type)
        return

    if return_type == 'Nil':
        return None

    return pyarg
