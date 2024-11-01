ctypedef fused pycallable_ft:
    BoundExtensionMethod


cdef void _make_python_varcall(pycallable_ft method, const Variant **p_args, size_t p_count, Variant *r_ret,
                                      GDExtensionCallError *r_error) noexcept:
    """
    Implements GDExtension's 'call' logic when calling Python methods from the Engine
    """
    cdef int i
    cdef object args = PyTuple_New(p_count)
    cdef Variant arg
    cdef object value

    for i in range(p_count):
        arg = deref(<Variant *>p_args[i])
        value = arg.pythonize()
        ref.Py_INCREF(value)
        PyTuple_SET_ITEM(args, i, value)

    cdef object ret = method(*args)
    if r_error:
        r_error[0].error = GDEXTENSION_CALL_OK

    type_funcs.variant_from_pyobject(ret, r_ret)


cdef void _make_python_ptrcall(pycallable_ft method, void *r_ret, const void **p_args, size_t p_count) noexcept:
    """
    Implements GDExtension's 'ptrcall' logic when calling Python methods from the Engine
    """
    cdef int8_t *type_info = method._type_info_opt
    cdef size_t i = 0, expected_size = len(method.type_info) - 1

    if p_count != expected_size:
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__, p_count,
                expected_size, method.type_info[1:], method.type_info[0])
        )
        UtilityFunctions.printerr(msg)
        raise GDExtensionPythonPtrCallError(msg)

    cdef object args = PyTuple_New(p_count)
    cdef object value = None

    cdef int8_t arg_type = ARGTYPE_NO_ARGTYPE
    for i in range(p_count):
        arg_type = type_info[i + 1]

        # NOTE: Cython compiles this to C switch/case
        if arg_type == ARGTYPE_NIL:
            value = None
        elif arg_type == ARGTYPE_BOOL:
            value = type_funcs.bool_to_pyobject(deref(<uint8_t *>p_args[i]))
        elif arg_type == ARGTYPE_INT:
            value = type_funcs.int_to_pyobject(deref(<int64_t *>p_args[i]))
        elif arg_type == ARGTYPE_FLOAT:
            value = type_funcs.float_to_pyobject(deref(<double *>p_args[i]))
        elif arg_type == ARGTYPE_STRING:
            value = type_funcs.string_to_pyobject(deref(<String *>p_args[i]))
        elif arg_type == ARGTYPE_VECTOR2:
            value = type_funcs.vector2_to_pyobject(deref(<Vector2 *>p_args[i]))
        elif arg_type == ARGTYPE_VECTOR2I:
            value = type_funcs.vector2i_to_pyobject(deref(<Vector2i *>p_args[i]))
        elif arg_type == ARGTYPE_RECT2:
            value = type_funcs.rect2_to_pyobject(deref(<Rect2 *>p_args[i]))
        elif arg_type == ARGTYPE_RECT2I:
            value = type_funcs.rect2i_to_pyobject(deref(<Rect2i *>p_args[i]))
        elif arg_type == ARGTYPE_VECTOR3:
            value = type_funcs.vector3_to_pyobject(deref(<Vector3 *>p_args[i]))
        elif arg_type == ARGTYPE_VECTOR3I:
            value = type_funcs.vector3i_to_pyobject(deref(<Vector3i *>p_args[i]))
        elif arg_type == ARGTYPE_TRANSFORM2D:
            value = type_funcs.transform2d_to_pyobject(deref(<Transform2D *>p_args[i]))
        elif arg_type == ARGTYPE_VECTOR4:
            value = type_funcs.vector4_to_pyobject(deref(<Vector4 *>p_args[i]))
        elif arg_type == ARGTYPE_VECTOR4I:
            value = type_funcs.vector4i_to_pyobject(deref(<Vector4i *>p_args[i]))
        elif arg_type == ARGTYPE_PLANE:
            value = type_funcs.plane_to_pyobject(deref(<Plane *>p_args[i]))
        elif arg_type == ARGTYPE_QUATERNION:
            value = type_funcs.quaternion_to_pyobject(deref(<Quaternion *>p_args[i]))
        elif arg_type == ARGTYPE_AABB:
            value = type_funcs.aabb_to_pyobject(deref(<_AABB *>p_args[i]))
        elif arg_type == ARGTYPE_BASIS:
            value = type_funcs.basis_to_pyobject(deref(<Basis *>p_args[i]))
        elif arg_type == ARGTYPE_TRANSFORM3D:
            value = type_funcs.transform3d_to_pyobject(deref(<Transform3D *>p_args[i]))
        elif arg_type == ARGTYPE_PROJECTION:
            value = type_funcs.projection_to_pyobject(deref(<Projection *>p_args[i]))
        elif arg_type == ARGTYPE_COLOR:
            value = type_funcs.color_to_pyobject(deref(<Color *>p_args[i]))
        elif arg_type == ARGTYPE_STRING_NAME:
            value = type_funcs.string_name_to_pyobject(deref(<StringName *>p_args[i]))
        elif arg_type == ARGTYPE_NODE_PATH:
            value = type_funcs.node_path_to_pyobject(deref(<NodePath *>p_args[i]))
        elif arg_type == ARGTYPE_RID:
            value = type_funcs.rid_to_pyobject(deref(<_RID *>p_args[i]))
        elif arg_type == ARGTYPE_OBJECT:
            value = object_to_pyobject(deref(<void **>p_args[i]))
        elif arg_type == ARGTYPE_CALLABLE:
            value = type_funcs.callable_to_pyobject(deref(<GodotCppCallable *>p_args[i]))
        elif arg_type == ARGTYPE_SIGNAL:
            value = type_funcs.signal_to_pyobject(deref(<GodotCppSignal *>p_args[i]))
        elif arg_type == ARGTYPE_DICTIONARY:
            value = type_funcs.dictionary_to_pyobject(deref(<Dictionary *>p_args[i]))
        elif arg_type == ARGTYPE_ARRAY:
            value = type_funcs.array_to_pyobject(deref(<Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_BYTE_ARRAY:
            value = type_funcs.packed_byte_array_to_pyobject(deref(<PackedByteArray *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_INT32_ARRAY:
            value = type_funcs.packed_int32_array_to_pyobject(deref(<PackedInt32Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_INT64_ARRAY:
            value = type_funcs.packed_int64_array_to_pyobject(deref(<PackedInt64Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
            value = type_funcs.packed_float32_array_to_pyobject(deref(<PackedFloat32Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
            value = type_funcs.packed_float64_array_to_pyobject(deref(<PackedFloat64Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_STRING_ARRAY:
            value = type_funcs.packed_string_array_to_pyobject(deref(<PackedStringArray *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
            value = type_funcs.packed_vector2_array_to_pyobject(deref(<PackedVector2Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
            value = type_funcs.packed_vector3_array_to_pyobject(deref(<PackedVector3Array *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_COLOR_ARRAY:
            value = type_funcs.packed_color_array_to_pyobject(deref(<PackedColorArray *>p_args[i]))
        elif arg_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
            value = type_funcs.packed_vector4_array_to_pyobject(deref(<PackedVector4Array *>p_args[i]))
        elif arg_type == ARGTYPE_VARIANT:
            value = type_funcs.variant_to_pyobject(deref(<Variant *>p_args[i]))
        elif arg_type == ARGTYPE_POINTER:
            value = type_funcs.pointer_to_pyobject(deref(<void **>p_args[i]))
        elif arg_type == ARGTYPE_AUDIO_FRAME:
            value = type_funcs.audio_frame_to_pyobject(deref(<AudioFrame **>p_args[i]))
        elif arg_type == ARGTYPE_CARET_INFO:
            value = type_funcs.caret_info_to_pyobject(deref(<CaretInfo **>p_args[i]))
        elif arg_type == ARGTYPE_GLYPH:
            value = type_funcs.glyph_to_pyobject(deref(<Glyph **>p_args[i]))
        elif arg_type == ARGTYPE_OBJECT_ID:
            value = type_funcs.object_id_to_pyobject(deref(<ObjectID **>p_args[i]))
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_MOTION_RESULT:
            value = type_funcs.physics_server2d_extension_motion_result_to_pyobject(
                deref(<PhysicsServer2DExtensionMotionResult **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_RAY_RESULT:
            value = type_funcs.physics_server2d_extension_ray_result_to_pyobject(
                deref(<PhysicsServer2DExtensionRayResult **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_REST_INFO:
            value = type_funcs.physics_server2d_extension_shape_rest_info_to_pyobject(
                deref(<PhysicsServer2DExtensionShapeRestInfo **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_RESULT:
            value = type_funcs.physics_server2d_extension_shape_result_to_pyobject(
                deref(<PhysicsServer2DExtensionShapeResult **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_COLLISION:
            value = type_funcs.physics_server3d_extension_motion_collision_to_pyobject(
                deref(<PhysicsServer3DExtensionMotionCollision **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_RESULT:
            value = type_funcs.physics_server3d_extension_motion_result_to_pyobject(
                deref(<PhysicsServer3DExtensionMotionResult **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_RAY_RESULT:
            value = type_funcs.physics_server3d_extension_ray_result_to_pyobject(
                deref(<PhysicsServer3DExtensionRayResult **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_REST_INFO:
            value = type_funcs.physics_server3d_extension_shape_rest_info_to_pyobject(
                deref(<PhysicsServer3DExtensionShapeRestInfo **>p_args[i])
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_RESULT:
            value = type_funcs.physics_server3d_extension_shape_result_to_pyobject(
                deref(<PhysicsServer3DExtensionShapeResult **>p_args[i])
            )
        elif arg_type == ARGTYPE_SCRIPTING_LANGUAGE_PROFILING_INFO:
            value = type_funcs.script_language_extension_profiling_info_to_pyobject(
                deref(<const ScriptLanguageExtensionProfilingInfo **>p_args[i])
            )
        else:
            msg = "Could not convert argument '%s[#%d]' in %r" % (method.type_info[i], arg_type, method)

            raise GDExtensionPythonPtrCallError(msg)

        ref.Py_INCREF(value)
        PyTuple_SET_ITEM(args, i, value)


    value = method(*args)
    cdef int8_t return_type = type_info[0]
    cdef size_t return_size = get_max_arg_size(type_info, 1)
    cdef numpy.ndarray return_value
    cdef void *ret_value_ptr = NULL

    if return_size > 0:
        return_value = np.array([b'\0' * return_size], dtype=np.void(return_size))
        ret_value_ptr = numpy.PyArray_GETPTR1(return_value, 0)
    else:
        if return_type != ARGTYPE_NIL:
            raise GDExtensionPythonPtrCallError("Attempt to return a value of zero size")

    # NOTE: Cython compiles this to C switch/case
    if return_type == ARGTYPE_NIL:
        pass
    elif return_type == ARGTYPE_BOOL:
        type_funcs.bool_from_pyobject(value, <uint8_t *>ret_value_ptr)
    elif return_type == ARGTYPE_INT:
        type_funcs.int_from_pyobject(value, <int64_t *>ret_value_ptr)
    elif return_type == ARGTYPE_FLOAT:
        type_funcs.float_from_pyobject(value, <double *>ret_value_ptr)
    elif return_type == ARGTYPE_STRING:
        type_funcs.string_from_pyobject(value, <String *>ret_value_ptr)
    elif return_type == ARGTYPE_VECTOR2:
        type_funcs.vector2_from_pyobject(value, <Vector2 *>ret_value_ptr)
    elif return_type == ARGTYPE_VECTOR2I:
        type_funcs.vector2i_from_pyobject(value, <Vector2i *>ret_value_ptr)
    elif return_type == ARGTYPE_RECT2:
        type_funcs.rect2_from_pyobject(value, <Rect2 *>ret_value_ptr)
    elif return_type == ARGTYPE_RECT2I:
        type_funcs.rect2i_from_pyobject(value, <Rect2i *>ret_value_ptr)
    elif return_type == ARGTYPE_VECTOR3:
        type_funcs.vector3_from_pyobject(value, <Vector3 *>ret_value_ptr)
    elif return_type == ARGTYPE_VECTOR3I:
        type_funcs.vector3i_from_pyobject(value, <Vector3i *>ret_value_ptr)
    elif return_type == ARGTYPE_TRANSFORM2D:
        type_funcs.transform2d_from_pyobject(value, <Transform2D *>ret_value_ptr)
    elif return_type == ARGTYPE_VECTOR4:
        type_funcs.vector4_from_pyobject(value, <Vector4 *>ret_value_ptr)
    elif return_type == ARGTYPE_VECTOR4I:
        type_funcs.vector4i_from_pyobject(value, <Vector4i *>ret_value_ptr)
    elif return_type == ARGTYPE_PLANE:
        type_funcs.plane_from_pyobject(value, <Plane *>ret_value_ptr)
    elif return_type == ARGTYPE_QUATERNION:
        type_funcs.quaternion_from_pyobject(value, <Quaternion *>ret_value_ptr)
    elif return_type == ARGTYPE_AABB:
        type_funcs.aabb_from_pyobject(value, <_AABB *>ret_value_ptr)
    elif return_type == ARGTYPE_BASIS:
        type_funcs.basis_from_pyobject(value, <Basis *>ret_value_ptr)
    elif return_type == ARGTYPE_TRANSFORM3D:
        type_funcs.transform3d_from_pyobject(value, <Transform3D *>ret_value_ptr)
    elif return_type == ARGTYPE_PROJECTION:
        type_funcs.projection_from_pyobject(value, <Projection *>ret_value_ptr)
    elif return_type == ARGTYPE_COLOR:
        type_funcs.color_from_pyobject(value, <Color *>ret_value_ptr)
    elif return_type == ARGTYPE_STRING_NAME:
        type_funcs.string_name_from_pyobject(value, <StringName *>ret_value_ptr)
    elif return_type == ARGTYPE_NODE_PATH:
        type_funcs.node_path_from_pyobject(value, <NodePath *>ret_value_ptr)
    elif return_type == ARGTYPE_RID:
        type_funcs.rid_from_pyobject(value, <_RID *>ret_value_ptr)
    elif return_type == ARGTYPE_OBJECT:
        object_from_pyobject(value, <void **>ret_value_ptr)
    elif return_type == ARGTYPE_CALLABLE:
        type_funcs.callable_from_pyobject(value, <GodotCppCallable *>ret_value_ptr)
    elif return_type == ARGTYPE_SIGNAL:
        type_funcs.signal_from_pyobject(value, <GodotCppSignal *>ret_value_ptr)
    elif return_type == ARGTYPE_DICTIONARY:
        type_funcs.dictionary_from_pyobject(value, <Dictionary *>ret_value_ptr)
    elif return_type == ARGTYPE_ARRAY:
        type_funcs.array_from_pyobject(value, <Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_BYTE_ARRAY:
        type_funcs.packed_byte_array_from_pyobject(value, <PackedByteArray *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_INT32_ARRAY:
        type_funcs.packed_int32_array_from_pyobject(value, <PackedInt32Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_INT64_ARRAY:
        type_funcs.packed_int64_array_from_pyobject(value, <PackedInt64Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
        type_funcs.packed_float32_array_from_pyobject(value, <PackedFloat32Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
        type_funcs.packed_float64_array_from_pyobject(value, <PackedFloat64Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_STRING_ARRAY:
        type_funcs.packed_string_array_from_pyobject(value, <PackedStringArray *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
        type_funcs.packed_vector2_array_from_pyobject(value, <PackedVector2Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
        type_funcs.packed_vector3_array_from_pyobject(value, <PackedVector3Array *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_COLOR_ARRAY:
        type_funcs.packed_color_array_from_pyobject(value, <PackedColorArray *>ret_value_ptr)
    elif return_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
        type_funcs.packed_vector4_array_from_pyobject(value, <PackedVector4Array *>ret_value_ptr)
    elif return_type == ARGTYPE_VARIANT:
        type_funcs.variant_from_pyobject(value, <Variant *>ret_value_ptr)
    elif return_type == ARGTYPE_POINTER:
        type_funcs.pointer_from_pyobject(value, <void **>ret_value_ptr)
    elif return_type == ARGTYPE_AUDIO_FRAME:
        type_funcs.audio_frame_from_pyobject(value, <AudioFrame *>ret_value_ptr)
    elif return_type == ARGTYPE_CARET_INFO:
        type_funcs.caret_info_from_pyobject(value, <CaretInfo *>ret_value_ptr)
    elif return_type == ARGTYPE_GLYPH:
        type_funcs.glyph_from_pyobject(value, <Glyph *>ret_value_ptr)
    elif return_type == ARGTYPE_OBJECT_ID:
        type_funcs.object_id_from_pyobject(value, <ObjectID *>ret_value_ptr)
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_MOTION_RESULT:
        type_funcs.physics_server2d_extension_motion_result_from_pyobject(
            value,
            <PhysicsServer2DExtensionMotionResult *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_RAY_RESULT:
        type_funcs.physics_server2d_extension_ray_result_from_pyobject(
            value,
            <PhysicsServer2DExtensionRayResult *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_REST_INFO:
        type_funcs.physics_server2d_extension_shape_rest_info_from_pyobject(
            value,
            <PhysicsServer2DExtensionShapeRestInfo *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_RESULT:
        type_funcs.physics_server2d_extension_shape_result_from_pyobject(
            value,
            <PhysicsServer2DExtensionShapeResult *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_COLLISION:
        type_funcs.physics_server3d_extension_motion_collision_from_pyobject(
            value,
            <PhysicsServer3DExtensionMotionCollision *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_RESULT:
        type_funcs.physics_server3d_extension_motion_result_from_pyobject(
            value,
            <PhysicsServer3DExtensionMotionResult *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_RAY_RESULT:
        type_funcs.physics_server3d_extension_ray_result_from_pyobject(
            value,
            <PhysicsServer3DExtensionRayResult *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_REST_INFO:
        type_funcs.physics_server3d_extension_shape_rest_info_from_pyobject(
            value,
            <PhysicsServer3DExtensionShapeRestInfo *>ret_value_ptr
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_RESULT:
        type_funcs.physics_server3d_extension_shape_result_from_pyobject(
            value,
            <PhysicsServer3DExtensionShapeResult *>ret_value_ptr
        )
    elif return_type == ARGTYPE_SCRIPTING_LANGUAGE_PROFILING_INFO:
        type_funcs.script_language_extension_profiling_info_from_pyobject(
            value,
            <ScriptLanguageExtensionProfilingInfo *>ret_value_ptr
        )
    else:
        msg = "Could not convert return value %r from %r in %r" % (method.type_info[0], value, method)
        UtilityFunctions.push_error(msg)

        raise GDExtensionPythonPtrCallError(msg)

    for i in range(return_size):
        (<uint8_t *>r_ret)[i] = (<uint8_t *>ret_value_ptr)[i]
