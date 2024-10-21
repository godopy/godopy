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


cdef void _make_python_varcall(pycallable_ft method, const void **p_args, size_t p_count, void *r_ret,
                               GDExtensionCallError *r_error) noexcept:
    """
    Implements GDExtension's 'call' logic when calling Python methods from the Engine
    """
    cdef int i
    cdef list args = []
    cdef Variant arg

    for i in range(p_count):
        arg = deref(<Variant *>p_args[i])
        args.append(arg.pythonize())

    cdef object ret = method(*args)
    if r_error:
        r_error[0].error = GDEXTENSION_CALL_OK

    (<Variant *>r_ret)[0] = Variant(<const PyObject *>ret)


cdef void _make_python_ptrcall(pycallable_ft method, void *r_ret, const void **p_args, size_t p_count) noexcept:
    """
    Implements GDExtension's 'ptrcall' logic when calling Python methods from the Engine
    """
    cdef tuple type_info = method.type_info
    cdef size_t i = 0, size = p_count

    # cdef size_t size = func.__code__.co_argcount - 1
    if size != (len(type_info) - 1):
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__, size, len(type_info) - 1, type_info[1:], type_info[0])
        )
        UtilityFunctions.printerr(msg)
        raise TypeError(msg)

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

    cdef PythonObject *python_object_arg

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

    cdef object result = method(*args)

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
        variant_arg = Variant(<const PyObject *>result)
        array_arg = <Array>variant_arg
        (<Array *>r_ret)[0] = array_arg
    elif return_type == 'Dictionary':
        variant_arg = Variant(<const PyObject *>result)
        dictionary_arg = <Dictionary>variant_arg
        (<Dictionary *>r_ret)[0] = dictionary_arg
    elif return_type == 'Variant':
        variant_arg = Variant(<const PyObject *>result)
        (<Variant *>r_ret)[0] = variant_arg
    elif return_type in _global_inheritance_info and isinstance(result, Object):
        object_arg = <Object>result
        (<void **>r_ret)[0] = object_arg._owner
    elif return_type == 'Object' and result is not None:
        python_object_arg = PythonRuntime.get_singleton().python_object_from_pyobject(result)
        (<void **>r_ret)[0] = python_object_arg._owner
    elif return_type in _global_inheritance_info and result is None:
        UtilityFunctions.push_warning("Expected %r but %r returned %r" % (return_type, method, result))

    elif return_type != 'Nil':
        if return_type in _global_inheritance_info:
            UtilityFunctions.push_error(
                "NOT IMPLEMENTED: Can't convert %r from %r in %r" % (return_type, result, method)
            )

        else:
            UtilityFunctions.push_error(
                "NOT IMPLEMENTED: "
                ("Can't convert %r return types in virtual functions yet. Result was: %r in function %r"
                    % (return_type, result, method))
            )
