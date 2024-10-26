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

    # Optimized get_node for Python nodes
    if method.__name__ == 'get_node' and size == 1 and args[0] in _NODEDB:
        pyarg = _NODEDB[args[0]]
        return pyarg

    cdef bint unknown_argtype_error = False
    cdef int8_t arg_type

    for i in range(size):
        arg_type = type_info[i + 1]
        pyarg = args[i]

        # NOTE: Cython compiles this to C switch/case
        if arg_type == ARGTYPE_BOOL:
            type_funcs.bool_from_pyobject(pyarg, &bool_arg)
            ptr_args[i] = &bool_arg
        elif arg_type == ARGTYPE_INT:
            type_funcs.int_from_pyobject(pyarg, &int_arg)
            ptr_args[i] = &int_arg
        elif arg_type == ARGTYPE_FLOAT:
            type_funcs.float_from_pyobject(pyarg, &float_arg)
            ptr_args[i] = &float_arg
        elif arg_type == ARGTYPE_STRING:
            type_funcs.string_from_pyobject(pyarg, &string_arg)
            ptr_args[i] = &string_arg
        elif arg_type == ARGTYPE_VECTOR2:
            type_funcs.vector2_from_pyobject(pyarg, &vector2_arg)
            ptr_args[i] = &vector2_arg
        elif arg_type == ARGTYPE_VECTOR2I:
            type_funcs.vector2i_from_pyobject(pyarg, &vector2i_arg)
            ptr_args[i] = &vector2i_arg
        elif arg_type == ARGTYPE_RECT2:
            type_funcs.rect2_from_pyobject(pyarg, &rect2_arg)
            ptr_args[i] = &rect2_arg
        elif arg_type == ARGTYPE_RECT2I:
            type_funcs.rect2i_from_pyobject(pyarg, &rect2i_arg)
            ptr_args[i] = &rect2i_arg
        elif arg_type == ARGTYPE_VECTOR3:
            type_funcs.vector3_from_pyobject(pyarg, &vector3_arg)
            ptr_args[i] = &vector3_arg
        elif arg_type == ARGTYPE_VECTOR3I:
            type_funcs.vector3i_from_pyobject(pyarg, &vector3i_arg)
            ptr_args[i] = &vector3i_arg
        elif arg_type == ARGTYPE_TRANSFORM2D:
            type_funcs.transform2d_from_pyobject(pyarg, &transform2d_arg)
            ptr_args[i] = &transform2d_arg
        elif arg_type == ARGTYPE_VECTOR4:
            type_funcs.vector4_from_pyobject(pyarg, &vector4_arg)
            ptr_args[i] = &vector4_arg
        elif arg_type == ARGTYPE_VECTOR4I:
            type_funcs.vector4i_from_pyobject(pyarg, &vector4i_arg)
            ptr_args[i] = &vector4i_arg
        elif arg_type == ARGTYPE_PLANE:
            type_funcs.plane_from_pyobject(pyarg, &plane_arg)
            ptr_args[i] = &plane_arg
        elif arg_type == ARGTYPE_QUATERNION:
            type_funcs.quaternion_from_pyobject(pyarg, &quaternion_arg)
            ptr_args[i] = &quaternion_arg
        elif arg_type == ARGTYPE_AABB:
            type_funcs.aabb_from_pyobject(pyarg, &aabb_arg)
            ptr_args[i] = &aabb_arg
        elif arg_type == ARGTYPE_BASIS:
            type_funcs.basis_from_pyobject(pyarg, &basis_arg)
            ptr_args[i] = &basis_arg
        elif arg_type == ARGTYPE_TRANSFORM3D:
            type_funcs.transform3d_from_pyobject(pyarg, &transform3d_arg)
            ptr_args[i] = &transform3d_arg
        elif arg_type == ARGTYPE_PROJECTION:
            type_funcs.projection_from_pyobject(pyarg, &projection_arg)
            ptr_args[i] = &projection_arg
        elif arg_type == ARGTYPE_COLOR:
            type_funcs.color_from_pyobject(pyarg, &color_arg)
            ptr_args[i] = &color_arg
        elif arg_type == ARGTYPE_STRING_NAME:
            type_funcs.string_name_from_pyobject(pyarg, &stringname_arg)
            ptr_args[i] = &stringname_arg
        elif arg_type == ARGTYPE_NODE_PATH:
            type_funcs.node_path_from_pyobject(pyarg, &nodepath_arg)
            ptr_args[i] = &nodepath_arg
        elif arg_type == ARGTYPE_RID:
            type_funcs.rid_from_pyobject(pyarg, &rid_arg)
            ptr_args[i] = &rid_arg
        elif arg_type == ARGTYPE_OBJECT:
            object_from_pyobject(pyarg, &ptr_arg)
            ptr_args[i] = &ptr_arg
        elif arg_type == ARGTYPE_CALLABLE:
            type_funcs.callable_from_pyobject(pyarg, &callable_arg)
            ptr_args[i] = &callable_arg
        elif arg_type == ARGTYPE_SIGNAL:
            type_funcs.signal_from_pyobject(pyarg, &signal_arg)
            ptr_args[i] = &signal_arg
        elif arg_type == ARGTYPE_DICTIONARY:
            type_funcs.dictionary_from_pyobject(pyarg, &dictionary_arg)
            ptr_args[i] = &dictionary_arg
        elif arg_type == ARGTYPE_ARRAY:
            type_funcs.array_from_pyobject(pyarg, &array_arg)
            ptr_args[i] = &array_arg
        elif arg_type == ARGTYPE_PACKED_BYTE_ARRAY:
            type_funcs.packed_byte_array_from_pyobject(pyarg, &packed_byte_array_arg)
            ptr_args[i] = &packed_byte_array_arg
        elif arg_type == ARGTYPE_PACKED_INT32_ARRAY:
            type_funcs.packed_int32_array_from_pyobject(pyarg, &packed_int32_array_arg)
            ptr_args[i] = &packed_int32_array_arg
        elif arg_type == ARGTYPE_PACKED_INT64_ARRAY:
            type_funcs.packed_int64_array_from_pyobject(pyarg, &packed_int64_array_arg)
            ptr_args[i] = &packed_int64_array_arg
        elif arg_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
            type_funcs.packed_float32_array_from_pyobject(pyarg, &packed_float32_array_arg)
            ptr_args[i] = &packed_float32_array_arg
        elif arg_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
            type_funcs.packed_float64_array_from_pyobject(pyarg, &packed_float64_array_arg)
            ptr_args[i] = &packed_float64_array_arg
        elif arg_type == ARGTYPE_PACKED_STRING_ARRAY:
            type_funcs.packed_string_array_from_pyobject(pyarg, &packed_string_array_arg)
            ptr_args[i] = &packed_string_array_arg
        elif arg_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
            type_funcs.packed_vector2_array_from_pyobject(pyarg, &packed_vector2_array_arg)
            ptr_args[i] = &packed_vector2_array_arg
        elif arg_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
            type_funcs.packed_vector3_array_from_pyobject(pyarg, &packed_vector3_array_arg)
            ptr_args[i] = &packed_vector3_array_arg
        elif arg_type == ARGTYPE_PACKED_COLOR_ARRAY:
            type_funcs.packed_color_array_from_pyobject(pyarg, &packed_color_array_arg)
            ptr_args[i] = &packed_color_array_arg
        elif arg_type == ARGTYPE_VARIANT:
            type_funcs.variant_from_pyobject(pyarg, &variant_arg)
            ptr_args[i] = &variant_arg
        else:
            unknown_argtype_error = True
            break

    if unknown_argtype_error:
        gdextension_interface_mem_free(ptr_args)
        msg = "NOT IMPLEMENTED: Could not convert %r from %r in %r" % (arg_type, pyarg, method)
        UtilityFunctions.printerr(msg)
        raise NotImplementedError(msg)

    cdef int8_t return_type = type_info[0]

    # NOTE: Cython compiles this to C switch/case
    if return_type == ARGTYPE_NIL:
        ptrcall(method, NULL, <const void **>ptr_args, size)
    elif return_type == ARGTYPE_BOOL:
        ptrcall(method, &bool_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.bool_to_pyobject(bool_arg)
    elif return_type == ARGTYPE_INT:
        ptrcall(method, &int_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.int_to_pyobject(int_arg)
    elif return_type == ARGTYPE_FLOAT:
        ptrcall(method, &float_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.float_to_pyobject(float_arg)
    elif return_type == ARGTYPE_STRING:
        ptrcall(method, &string_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.string_to_pyobject(string_arg)
    elif return_type == ARGTYPE_VECTOR2:
        ptrcall(method, &vector2_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector2_to_pyobject(vector2_arg)
    elif return_type == ARGTYPE_VECTOR2I:
        ptrcall(method, &vector2i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector2i_to_pyobject(vector2i_arg)
    elif return_type == ARGTYPE_RECT2:
        ptrcall(method, &rect2_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rect2_to_pyobject(rect2_arg)
    elif return_type == ARGTYPE_RECT2I:
        ptrcall(method, &rect2i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rect2i_to_pyobject(rect2i_arg)
    elif return_type == ARGTYPE_VECTOR3:
        ptrcall(method, &vector3_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector3_to_pyobject(vector3_arg)
    elif return_type == ARGTYPE_VECTOR3I:
        ptrcall(method, &vector3i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector3i_to_pyobject(vector3i_arg)
    elif return_type == ARGTYPE_TRANSFORM2D:
        ptrcall(method, &transform2d_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.transform2d_to_pyobject(transform2d_arg)
    elif return_type == ARGTYPE_VECTOR4:
        ptrcall(method, &vector4_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector4_to_pyobject(vector4_arg)
    elif return_type == ARGTYPE_VECTOR4I:
        ptrcall(method, &vector4i_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.vector4i_to_pyobject(vector4i_arg)
    elif return_type == ARGTYPE_PLANE:
        ptrcall(method, &plane_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.plane_to_pyobject(plane_arg)
    elif return_type == ARGTYPE_QUATERNION:
        ptrcall(method, &quaternion_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.quaternion_to_pyobject(quaternion_arg)
    elif return_type == ARGTYPE_AABB:
        ptrcall(method, &aabb_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.aabb_to_pyobject(aabb_arg)
    elif return_type == ARGTYPE_BASIS:
        ptrcall(method, &basis_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.basis_to_pyobject(basis_arg)
    elif return_type == ARGTYPE_TRANSFORM3D:
        ptrcall(method, &transform3d_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.transform3d_to_pyobject(transform3d_arg)
    elif return_type == ARGTYPE_PROJECTION:
        ptrcall(method, &projection_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.projection_to_pyobject(projection_arg)
    elif return_type == ARGTYPE_COLOR:
        ptrcall(method, &color_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.color_to_pyobject(color_arg)
    elif return_type == ARGTYPE_STRING_NAME:
        ptrcall(method, &stringname_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.string_name_to_pyobject(stringname_arg)
    elif return_type == ARGTYPE_NODE_PATH:
        ptrcall(method, &nodepath_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.node_path_to_pyobject(nodepath_arg)
    elif return_type == ARGTYPE_RID:
        ptrcall(method, &rid_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.rid_to_pyobject(rid_arg)
    elif return_type == ARGTYPE_OBJECT:
        ptrcall(method, &ptr_arg, <const void **>ptr_args, size)
        pyarg = object_to_pyobject(ptr_arg)
    elif return_type == ARGTYPE_CALLABLE:
        ptrcall(method, &callable_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.callable_to_pyobject(callable_arg)
    elif return_type == ARGTYPE_SIGNAL:
        ptrcall(method, &signal_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.signal_to_pyobject(signal_arg)
    elif return_type == ARGTYPE_DICTIONARY:
        ptrcall(method, &dictionary_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.dictionary_to_pyobject(dictionary_arg)
    elif return_type == ARGTYPE_ARRAY:
        ptrcall(method, &array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.array_to_pyobject(array_arg)
    elif return_type == ARGTYPE_PACKED_BYTE_ARRAY:
        ptrcall(method, &packed_byte_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_byte_array_to_pyobject(packed_byte_array_arg)
    elif return_type == ARGTYPE_PACKED_INT32_ARRAY:
        ptrcall(method, &packed_int32_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_int32_array_to_pyobject(packed_int32_array_arg)
    elif return_type == ARGTYPE_PACKED_INT64_ARRAY:
        ptrcall(method, &packed_int64_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_int64_array_to_pyobject(packed_int64_array_arg)
    elif return_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
        ptrcall(method, &packed_float32_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_float32_array_to_pyobject(packed_float32_array_arg)
    elif return_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
        ptrcall(method, &packed_float64_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_float64_array_to_pyobject(packed_float64_array_arg)
    elif return_type == ARGTYPE_PACKED_STRING_ARRAY:
        ptrcall(method, &packed_string_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_string_array_to_pyobject(packed_string_array_arg)
    elif return_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
        ptrcall(method, &packed_vector2_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_vector2_array_to_pyobject(packed_vector2_array_arg)
    elif return_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
        ptrcall(method, &packed_vector3_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_vector3_array_to_pyobject(packed_vector3_array_arg)
    elif return_type == ARGTYPE_PACKED_COLOR_ARRAY:
        ptrcall(method, &packed_color_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_color_array_to_pyobject(packed_color_array_arg)
    elif return_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
        ptrcall(method, &packed_vector4_array_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.packed_vector4_array_to_pyobject(packed_vector4_array_arg)
    elif return_type == ARGTYPE_VARIANT:
        ptrcall(method, &variant_arg, <const void **>ptr_args, size)
        pyarg = type_funcs.variant_to_pyobject(variant_arg)
    else:
        unknown_argtype_error = True

    gdextension_interface_mem_free(ptr_args)

    if unknown_argtype_error:
        UtilityFunctions.push_error(
            "NOT IMPLEMENTED: Could not convert %r from %r in %r" % (return_type, pyarg, method)
        )
        return

    if return_type != ARGTYPE_NIL:
        return pyarg
