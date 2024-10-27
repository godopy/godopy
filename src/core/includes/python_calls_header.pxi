ctypedef fused pycallable_ft:
    BoundExtensionMethod


cdef inline void _make_python_varcall(pycallable_ft method, const Variant **p_args, size_t p_count, Variant *r_ret,
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


cdef inline void _make_python_ptrcall(pycallable_ft method, void *r_ret, const void **p_args, size_t p_count) noexcept:
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
    cdef Vector3 vector3_arg
    cdef Vector3i vector3i_arg
    cdef Transform2D transform2d_arg
    cdef Vector4 vector4_arg
    cdef Vector4i vector4i_arg
    cdef Plane plane_arg
    cdef Quaternion quaternion_arg
    cdef _AABB aabb_arg
    cdef Basis basis_arg
    cdef Transform3D transform3d_arg
    cdef Projection projection_arg
    cdef Color color_arg
    cdef StringName stringname_arg
    cdef NodePath nodepath_arg
    cdef _RID rid_arg
    cdef void *ptr_arg  # Object
    cdef GodotCppCallable callable_arg
    cdef GodotCppSignal signal_arg
    cdef Dictionary dictionary_arg
    cdef Array array_arg
    cdef PackedByteArray packed_byte_array_arg
    cdef PackedInt32Array packed_int32_array_arg
    cdef PackedInt64Array packed_int64_array_arg
    cdef PackedFloat32Array packed_float32_array_arg
    cdef PackedFloat64Array packed_float64_array_arg
    cdef PackedStringArray packed_string_array_arg
    cdef PackedVector2Array packed_vector2_array_arg
    cdef PackedVector3Array packed_vector3_array_arg
    cdef PackedColorArray packed_color_array_arg
    cdef PackedVector4Array packed_vector4_array_arg

    cdef Variant variant_arg
    cdef object pyarg

    cdef int8_t arg_type
    for i in range(p_count):
        arg_type = type_info[i + 1]

        # NOTE: Cython compiles this to C switch/case
        if arg_type == ARGTYPE_BOOL:
            pyarg = type_funcs.bool_to_pyobject(deref(<uint8_t *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_INT:
            pyarg = type_funcs.int_to_pyobject(deref(<int64_t *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_FLOAT:
            pyarg = type_funcs.float_to_pyobject(deref(<double *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_STRING:
            pyarg = type_funcs.string_to_pyobject(deref(<String *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VECTOR2:
            pyarg = type_funcs.vector2_to_pyobject(deref(<Vector2 *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VECTOR2I:
            pyarg = type_funcs.vector2i_to_pyobject(deref(<Vector2i *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_RECT2:
            pyarg = type_funcs.rect2_to_pyobject(deref(<Rect2 *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_RECT2I:
            pyarg = type_funcs.rect2i_to_pyobject(deref(<Rect2i *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VECTOR3:
            pyarg = type_funcs.vector3_to_pyobject(deref(<Vector3 *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VECTOR3I:
            pyarg = type_funcs.vector3i_to_pyobject(deref(<Vector3i *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_TRANSFORM2D:
            pyarg = type_funcs.transform2d_to_pyobject(deref(<Transform2D *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VECTOR4:
            pyarg = type_funcs.vector4_to_pyobject(deref(<Vector4 *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VECTOR4I:
            pyarg = type_funcs.vector4i_to_pyobject(deref(<Vector4i *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PLANE:
            pyarg = type_funcs.plane_to_pyobject(deref(<Plane *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_QUATERNION:
            pyarg = type_funcs.quaternion_to_pyobject(deref(<Quaternion *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_AABB:
            pyarg = type_funcs.aabb_to_pyobject(deref(<_AABB *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_BASIS:
            pyarg = type_funcs.basis_to_pyobject(deref(<Basis *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_TRANSFORM3D:
            pyarg = type_funcs.transform3d_to_pyobject(deref(<Transform3D *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PROJECTION:
            pyarg = type_funcs.projection_to_pyobject(deref(<Projection *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_COLOR:
            pyarg = type_funcs.color_to_pyobject(deref(<Color *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_STRING_NAME:
            pyarg = type_funcs.string_name_to_pyobject(deref(<StringName *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_NODE_PATH:
            pyarg = type_funcs.node_path_to_pyobject(deref(<NodePath *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_RID:
            pyarg = type_funcs.rid_to_pyobject(deref(<_RID *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_OBJECT:
            pyarg = object_to_pyobject(deref(<void **>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_CALLABLE:
            pyarg = type_funcs.callable_to_pyobject(deref(<GodotCppCallable *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_SIGNAL:
            pyarg = type_funcs.signal_to_pyobject(deref(<GodotCppSignal *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_DICTIONARY:
            pyarg = type_funcs.dictionary_to_pyobject(deref(<Dictionary *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_ARRAY:
            pyarg = type_funcs.array_to_pyobject(deref(<Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_BYTE_ARRAY:
            pyarg = type_funcs.packed_byte_array_to_pyobject(deref(<PackedByteArray *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_INT32_ARRAY:
            pyarg = type_funcs.packed_int32_array_to_pyobject(deref(<PackedInt32Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_INT64_ARRAY:
            pyarg = type_funcs.packed_int64_array_to_pyobject(deref(<PackedInt64Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
            pyarg = type_funcs.packed_float32_array_to_pyobject(deref(<PackedFloat32Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
            pyarg = type_funcs.packed_float64_array_to_pyobject(deref(<PackedFloat64Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_STRING_ARRAY:
            pyarg = type_funcs.packed_string_array_to_pyobject(deref(<PackedStringArray *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
            pyarg = type_funcs.packed_vector2_array_to_pyobject(deref(<PackedVector2Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
            pyarg = type_funcs.packed_vector3_array_to_pyobject(deref(<PackedVector3Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_COLOR_ARRAY:
            pyarg = type_funcs.packed_color_array_to_pyobject(deref(<PackedColorArray *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
            pyarg = type_funcs.packed_vector4_array_to_pyobject(deref(<PackedVector4Array *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        elif arg_type == ARGTYPE_VARIANT:
            pyarg = type_funcs.variant_to_pyobject(deref(<Variant *>p_args[i]))
            ref.Py_INCREF(pyarg)
            PyTuple_SET_ITEM(args, i, pyarg)
        else:
            UtilityFunctions.push_error(
                "NOT IMPLEMENTED: Can't convert %r arguments in Python ptrcalls" % arg_type
            )
            ref.Py_INCREF(None)
            PyTuple_SET_ITEM(args, i, None)

    pyarg = method(*args)
    cdef int8_t return_type = type_info[0]

    # NOTE: Cython compiles this to C switch/case
    if return_type == ARGTYPE_BOOL:
        type_funcs.bool_from_pyobject(pyarg, &bool_arg)
        (<bint *>r_ret)[0] = bool_arg
    elif return_type == ARGTYPE_INT:
        type_funcs.int_from_pyobject(pyarg, &int_arg)
        (<int64_t *>r_ret)[0] = int_arg
    elif return_type == ARGTYPE_FLOAT:
        type_funcs.float_from_pyobject(pyarg, &float_arg)
        (<double *>r_ret)[0] = float_arg
    elif return_type == ARGTYPE_STRING:
        type_funcs.string_from_pyobject(pyarg, &string_arg)
        (<String *>r_ret)[0] = string_arg
    elif return_type == ARGTYPE_VECTOR2:
        type_funcs.vector2_from_pyobject(pyarg, &vector2_arg)
        (<Vector2 *>r_ret)[0] = vector2_arg
    elif return_type == ARGTYPE_VECTOR2I:
        type_funcs.vector2i_from_pyobject(pyarg, &vector2i_arg)
        (<Vector2i *>r_ret)[0] = vector2i_arg
    elif return_type == ARGTYPE_RECT2:
        type_funcs.rect2_from_pyobject(pyarg, &rect2_arg)
        (<Rect2 *>r_ret)[0] = rect2_arg
    elif return_type == ARGTYPE_RECT2I:
        type_funcs.rect2i_from_pyobject(pyarg, &rect2i_arg)
        (<Rect2i *>r_ret)[0] = rect2i_arg
    elif return_type == ARGTYPE_VECTOR3:
        type_funcs.vector3_from_pyobject(pyarg, &vector3_arg)
        (<Vector3 *>r_ret)[0] = vector3_arg
    elif return_type == ARGTYPE_VECTOR3I:
        type_funcs.vector3i_from_pyobject(pyarg, &vector3i_arg)
        (<Vector3i *>r_ret)[0] = vector3i_arg
    elif return_type == ARGTYPE_TRANSFORM2D:
        type_funcs.transform2d_from_pyobject(pyarg, &transform2d_arg)
        (<Transform2D *>r_ret)[0] = transform2d_arg
    elif return_type == ARGTYPE_VECTOR4:
        type_funcs.vector4_from_pyobject(pyarg, &vector4_arg)
        (<Vector4 *>r_ret)[0] = vector4_arg
    elif return_type == ARGTYPE_VECTOR4I:
        type_funcs.vector4i_from_pyobject(pyarg, &vector4i_arg)
        (<Vector4i *>r_ret)[0] = vector4i_arg
    elif return_type == ARGTYPE_PLANE:
        type_funcs.plane_from_pyobject(pyarg, &plane_arg)
        (<Plane *>r_ret)[0] = plane_arg
    elif return_type == ARGTYPE_QUATERNION:
        type_funcs.quaternion_from_pyobject(pyarg, &quaternion_arg)
        (<Quaternion *>r_ret)[0] = quaternion_arg
    elif return_type == ARGTYPE_AABB:
        type_funcs.aabb_from_pyobject(pyarg, &aabb_arg)
        (<_AABB *>r_ret)[0] = aabb_arg
    elif return_type == ARGTYPE_BASIS:
        type_funcs.basis_from_pyobject(pyarg, &basis_arg)
        (<Basis *>r_ret)[0] = basis_arg
    elif return_type == ARGTYPE_TRANSFORM3D:
        type_funcs.transform3d_from_pyobject(pyarg, &transform3d_arg)
        (<Transform3D *>r_ret)[0] = transform3d_arg
    elif return_type == ARGTYPE_PROJECTION:
        type_funcs.projection_from_pyobject(pyarg, &projection_arg)
        (<Projection *>r_ret)[0] = projection_arg
    elif return_type == ARGTYPE_COLOR:
        type_funcs.color_from_pyobject(pyarg, &color_arg)
        (<Color *>r_ret)[0] = color_arg
    elif return_type == ARGTYPE_STRING_NAME:
        type_funcs.string_name_from_pyobject(pyarg, &stringname_arg)
        (<StringName *>r_ret)[0] = stringname_arg
    elif return_type == ARGTYPE_NODE_PATH:
        type_funcs.node_path_from_pyobject(pyarg, &nodepath_arg)
        (<NodePath *>r_ret)[0] = nodepath_arg
    elif return_type == ARGTYPE_RID:
        type_funcs.rid_from_pyobject(pyarg, &rid_arg)
        (<_RID *>r_ret)[0] = rid_arg
    elif return_type == ARGTYPE_OBJECT:
        object_from_pyobject(pyarg, &ptr_arg)
        (<void **>r_ret)[0] = ptr_arg
    elif return_type == ARGTYPE_CALLABLE:
        type_funcs.callable_from_pyobject(pyarg, &callable_arg)
        (<GodotCppCallable *>r_ret)[0] = callable_arg
    elif return_type == ARGTYPE_SIGNAL:
        type_funcs.signal_from_pyobject(pyarg, &signal_arg)
        (<GodotCppSignal *>r_ret)[0] = signal_arg
    elif return_type == ARGTYPE_DICTIONARY:
        type_funcs.dictionary_from_pyobject(pyarg, &dictionary_arg)
        (<Dictionary *>r_ret)[0] = dictionary_arg
    elif return_type == ARGTYPE_ARRAY:
        type_funcs.array_from_pyobject(pyarg, &array_arg)
        (<Array *>r_ret)[0] = array_arg
    elif return_type == ARGTYPE_PACKED_BYTE_ARRAY:
        type_funcs.packed_byte_array_from_pyobject(pyarg, &packed_byte_array_arg)
        (<PackedByteArray *>r_ret)[0] = packed_byte_array_arg
    elif return_type == ARGTYPE_PACKED_INT32_ARRAY:
        type_funcs.packed_int32_array_from_pyobject(pyarg, &packed_int32_array_arg)
        (<PackedInt32Array *>r_ret)[0] = packed_int32_array_arg
    elif return_type == ARGTYPE_PACKED_INT64_ARRAY:
        type_funcs.packed_int64_array_from_pyobject(pyarg, &packed_int64_array_arg)
        (<PackedInt64Array *>r_ret)[0] = packed_int64_array_arg
    elif return_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
        type_funcs.packed_float32_array_from_pyobject(pyarg, &packed_float32_array_arg)
        (<PackedFloat32Array *>r_ret)[0] = packed_float32_array_arg
    elif return_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
        type_funcs.packed_float64_array_from_pyobject(pyarg, &packed_float64_array_arg)
        (<PackedFloat64Array *>r_ret)[0] = packed_float64_array_arg
    elif return_type == ARGTYPE_PACKED_STRING_ARRAY:
        type_funcs.packed_string_array_from_pyobject(pyarg, &packed_string_array_arg)
        (<PackedStringArray *>r_ret)[0] = packed_string_array_arg
    elif return_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
        type_funcs.packed_vector2_array_from_pyobject(pyarg, &packed_vector2_array_arg)
        (<PackedVector2Array *>r_ret)[0] = packed_vector2_array_arg
    elif return_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
        type_funcs.packed_vector3_array_from_pyobject(pyarg, &packed_vector3_array_arg)
        (<PackedVector3Array *>r_ret)[0] = packed_vector3_array_arg
    elif return_type == ARGTYPE_PACKED_COLOR_ARRAY:
        type_funcs.packed_color_array_from_pyobject(pyarg, &packed_color_array_arg)
        (<PackedColorArray *>r_ret)[0] = packed_color_array_arg
    elif return_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
        type_funcs.packed_vector4_array_from_pyobject(pyarg, &packed_vector4_array_arg)
        (<PackedVector4Array *>r_ret)[0] = packed_vector4_array_arg
    elif return_type == ARGTYPE_VARIANT:
        type_funcs.variant_from_pyobject(pyarg, &variant_arg)
        (<Variant *>r_ret)[0] = variant_arg
    elif return_type != ARGTYPE_NIL:
        UtilityFunctions.push_error(
            "NOT IMPLEMENTED: Could not convert %r from %r in %r" % (return_type, pyarg, method)
        )
