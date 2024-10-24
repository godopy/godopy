ctypedef fused pycallable_ft:
    BoundExtensionMethod


cdef class PythonCallableBase:
    """
    Base class for BoundExtensionMethod and (TODO) CustomCallable.
    """
    def __cinit__(self, *args):
        self.type_info = ()
        self.__func__ = None
        self.__name__ = ''


    def __init__(self):
        raise NotImplementedError("Base class, cannot instantiate")


cdef void _make_python_varcall(pycallable_ft method, const Variant **p_args, size_t p_count, Variant *r_ret,
                               GDExtensionCallError *r_error) noexcept:
    """
    Implements GDExtension's 'call' logic when calling Python methods from the Engine
    """
    cdef int i
    cdef object args = PyTuple_New(p_count)
    cdef Variant arg
    cdef object pyarg

    for i in range(p_count):
        arg = deref(<Variant *>p_args[i])
        pyarg = arg.pythonize()
        ref.Py_INCREF(pyarg)
        PyTuple_SET_ITEM(args, i, pyarg)

    cdef object ret = method(*args)
    if r_error:
        r_error[0].error = GDEXTENSION_CALL_OK

    r_ret[0] = Variant(<const PyObject *>ret)


cdef void _make_python_ptrcall(pycallable_ft method, void *r_ret, const void **p_args, size_t p_count) noexcept:
    """
    Implements GDExtension's 'ptrcall' logic when calling Python methods from the Engine
    """
    cdef int8_t *type_info = method._type_info_opt
    cdef size_t i = 0

    if p_count != (len(method.type_info) - 1):
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__, p_count,
                len(method.type_info) - 1, method.type_info[1:], method.type_info[0])
        )
        UtilityFunctions.printerr(msg)
        raise TypeError(msg)

    cdef object args = PyTuple_New(p_count)

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

    cdef PackedStringArray packedstringarray_arg

    cdef object pyarg
    cdef Variant variant_arg

    cdef int8_t arg_type
    for i in range(p_count):
        arg_type = type_info[i + 1]

        # NOTE: Cython compiles this to C switch/case
        if arg_type == BOOL:
            pyarg = type_funcs.bool_to_pyobject(deref(<uint8_t *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == INT:
            pyarg = type_funcs.int_to_pyobject(deref(<int64_t *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == FLOAT:
            pyarg = type_funcs.float_to_pyobject(deref(<double *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == STRING:
            pyarg = type_funcs.string_to_pyobject(deref(<String *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == VECTOR2:
            pyarg = type_funcs.vector2_to_pyobject(deref(<Vector2 *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == VECTOR2I:
            pyarg = type_funcs.vector2i_to_pyobject(deref(<Vector2i *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == RECT2:
            pyarg = type_funcs.rect2_to_pyobject(deref(<Rect2 *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == RECT2I:
            pyarg = type_funcs.rect2i_to_pyobject(deref(<Rect2i *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == STRING_NAME:
            pyarg = type_funcs.string_name_to_pyobject(deref(<StringName *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == NODE_PATH:
            pyarg = type_funcs.node_path_to_pyobject(deref(<NodePath *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == RID:
            pyarg = type_funcs.rid_to_pyobject(deref(<_RID *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == OBJECT:
            pyarg = object_to_pyobject(deref(<void **>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == DICTIONARY:
            pyarg = type_funcs.dictionary_to_pyobject(deref(<Dictionary *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARRAY:
            pyarg = type_funcs.array_to_pyobject(deref(<Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == PACKED_STRING_ARRAY:
            pyarg = type_funcs.packed_string_array_to_pyobject(deref(<PackedStringArray *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == VARIANT_MAX:
            pyarg = type_funcs.variant_to_pyobject(deref(<Variant *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        else:
            UtilityFunctions.push_error(
                "NOT IMPLEMENTED: Can't convert %r arguments in Python ptrcalls" % arg_type
            )
            ref.Py_INCREF(None)
            PyTuple_SET_ITEM(args, i, None)

    cdef object result = method(*args)

    cdef int return_type = type_info[0]

    # NOTE: Cython compiles this to C switch/case
    if return_type == BOOL:
        type_funcs.bool_from_pyobject(result, &bool_arg)
        (<bint *>r_ret)[0] = bool_arg
    elif return_type == INT:
        type_funcs.int_from_pyobject(result, &int_arg)
        (<int64_t *>r_ret)[0] = int_arg
    elif return_type == FLOAT:
        type_funcs.float_from_pyobject(result, &float_arg)
        (<double *>r_ret)[0] = float_arg
    elif return_type == STRING:
        type_funcs.string_from_pyobject(result, &string_arg)
        (<String *>r_ret)[0] = string_arg
    elif return_type == VECTOR2:
        type_funcs.vector2_from_pyobject(result, &vector2_arg)
        (<Vector2 *>r_ret)[0] = vector2_arg
    elif return_type == VECTOR2I:
        type_funcs.vector2i_from_pyobject(result, &vector2i_arg)
        (<Vector2 *>r_ret)[0] = vector2_arg
    elif return_type == RECT2:
        type_funcs.rect2_from_pyobject(result, &rect2_arg)
        (<Rect2 *>r_ret)[0] = rect2_arg
    elif return_type == RECT2I:
        type_funcs.rect2i_from_pyobject(result, &rect2i_arg)
        (<Rect2i *>r_ret)[0] = rect2i_arg
    elif return_type == STRING_NAME:
        type_funcs.string_name_from_pyobject(result, &stringname_arg)
        (<StringName *>r_ret)[0] = stringname_arg
    elif return_type == NODE_PATH:
        type_funcs.node_path_from_pyobject(result, &nodepath_arg)
        (<NodePath *>r_ret)[0] = nodepath_arg
    elif return_type == RID:
        type_funcs.rid_from_pyobject(result, &rid_arg)
        (<_RID *>r_ret)[0] = rid_arg
    elif return_type == OBJECT:
        object_from_pyobject(result, &ptr_arg)
        (<void **>r_ret)[0] = ptr_arg
    elif return_type == DICTIONARY:
        type_funcs.dictionary_from_pyobject(result, &dictionary_arg)
        (<Dictionary *>r_ret)[0] = dictionary_arg
    elif return_type == ARRAY:
        type_funcs.array_from_pyobject(result, &array_arg)
        (<Array *>r_ret)[0] = array_arg
    elif return_type == PACKED_STRING_ARRAY:
        type_funcs.packed_string_array_from_pyobject(result, &packedstringarray_arg)
        (<PackedStringArray *>r_ret)[0] = packedstringarray_arg
    elif return_type == VARIANT_MAX:
        variant_arg = Variant(<const PyObject *>result)
        (<Variant *>r_ret)[0] = variant_arg
    elif return_type != NIL:
        if return_type == OBJECT:
            UtilityFunctions.push_error(
                "NOT IMPLEMENTED: Can't convert %r from %r in %r" % (return_type, result, method)
            )

        else:
            UtilityFunctions.push_error(
                "NOT IMPLEMENTED: "
                ("Can't convert %r return types Python ptr calls. Result was: %r in function %r"
                    % (return_type, result, method))
            )
