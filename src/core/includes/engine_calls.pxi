ctypedef fused ptrcallable_t:
    MethodBind
    UtilityFunction
    BuiltinMethod


ctypedef fused varcallable_t:
    VariantMethod
    VariantStaticMethod
    MethodBind
    ScriptMethod


ctypedef void (*_ptrcall_func)(ptrcallable_t, void *, const void **, size_t) noexcept nogil
ctypedef void (*_varcall_func)(varcallable_t, const Variant **, size_t, Variant *, GDExtensionCallError *) noexcept nogil


@cython.final
cdef class _VariantPtrArray:
    cdef vector[Variant] args
    cdef size_t count
    cdef _Memory memory

    def __cinit__(self, object args):
        cdef size_t i
        self.count = len(args)

        self.args = vector[Variant](self.count)
        self.memory = _Memory(self.count * cython.sizeof(GDExtensionVariantPtr))

        for i in range(self.count):
            type_funcs.variant_from_pyobject(args[i], &self.args[i])
            (<Variant **>self.memory.ptr)[i] = &self.args[i]

    def __dealloc__(self):
        self.memory.free()

    cdef const Variant **ptr(self):
        return <const Variant **>self.memory.ptr


cdef object _make_engine_varcall(varcallable_t method, _varcall_func varcall, object args):
    """
    Implements GDExtension's 'call' logic when calling Engine methods from Python
    """
    cdef Variant return_value
    cdef GDExtensionCallError err

    err.error = GDEXTENSION_CALL_OK

    cdef _VariantPtrArray vargs = _VariantPtrArray(args)
    varcall(method, vargs.ptr(), vargs.count, &return_value, &err)

    if err.error != GDEXTENSION_CALL_OK:
        error_text = type_funcs.variant_to_pyobject(return_value)

        raise GDExtensionCallException(error_text, <int>err.error)

    return type_funcs.variant_to_pyobject(return_value)


cdef object _make_engine_ptrcall(ptrcallable_t method, _ptrcall_func ptrcall, object args):
    """
    Implements GDExtension's 'ptrcall' logic when calling Engine methods from Python
    """
    cdef int8_t *type_info = method._type_info_opt
    cdef size_t i = 0, size = len(args), expected_size = len(method.type_info) - 1

    if size != expected_size:
        msg = (
            '%s %s: wrong number of arguments: %d, %d expected. Arg types: %r. Return type: %r'
                % (method.__class__.__name__, method.__name__,  size, expected_size,
                    method.type_info[1:], method.type_info[0])
        )
        raise GDExtensionEnginePtrCallError(msg)

    cdef _Memory args_mem = _Memory(size * cython.sizeof(GDExtensionConstTypePtr))
    cdef object value

    # Optimized get_node for Python nodes
    if method.__name__ == 'get_node' and size == 1 and args[0] in _NODEDB:
        value = _NODEDB[args[0]]
        return value

    cdef size_t max_size = get_max_arg_size(type_info, size + 1)
    cdef numpy.ndarray arg_values

    if max_size > 0:
        arg_values = np.array([b'\0' * max_size] * (size + 1), dtype=np.void(max_size))

    cdef int8_t arg_type = ARGTYPE_NO_ARGTYPE
    cdef void *arg_value_ptr

    for i in range(size):
        arg_type = type_info[i + 1]
        value = args[i]
        arg_value_ptr = numpy.PyArray_GETPTR1(arg_values, i + 1)

        # NOTE: Cython compiles this to C switch/case
        if arg_type == ARGTYPE_NIL:
            (<void **>arg_value_ptr)[0] = NULL
        elif arg_type == ARGTYPE_BOOL:
            type_funcs.bool_from_pyobject(value, <uint8_t *>arg_value_ptr)
        elif arg_type == ARGTYPE_INT:
            type_funcs.int_from_pyobject(value, <int64_t *>arg_value_ptr)
        elif arg_type == ARGTYPE_FLOAT:
            type_funcs.float_from_pyobject(value, <double *>arg_value_ptr)
        elif arg_type == ARGTYPE_STRING:
            type_funcs.string_from_pyobject(value, <String *>arg_value_ptr)
        elif arg_type == ARGTYPE_VECTOR2:
            type_funcs.vector2_from_pyobject(value, <Vector2 *>arg_value_ptr)
        elif arg_type == ARGTYPE_VECTOR2I:
            type_funcs.vector2i_from_pyobject(value, <Vector2i *>arg_value_ptr)
        elif arg_type == ARGTYPE_RECT2:
            type_funcs.rect2_from_pyobject(value, <Rect2 *>arg_value_ptr)
        elif arg_type == ARGTYPE_RECT2I:
            type_funcs.rect2i_from_pyobject(value, <Rect2i *>arg_value_ptr)
        elif arg_type == ARGTYPE_VECTOR3:
            type_funcs.vector3_from_pyobject(value, <Vector3 *>arg_value_ptr)
        elif arg_type == ARGTYPE_VECTOR3I:
            type_funcs.vector3i_from_pyobject(value, <Vector3i *>arg_value_ptr)
        elif arg_type == ARGTYPE_TRANSFORM2D:
            type_funcs.transform2d_from_pyobject(value, <Transform2D *>arg_value_ptr)
        elif arg_type == ARGTYPE_VECTOR4:
            type_funcs.vector4_from_pyobject(value, <Vector4 *>arg_value_ptr)
        elif arg_type == ARGTYPE_VECTOR4I:
            type_funcs.vector4i_from_pyobject(value, <Vector4i *>arg_value_ptr)
        elif arg_type == ARGTYPE_PLANE:
            type_funcs.plane_from_pyobject(value, <Plane *>arg_value_ptr)
        elif arg_type == ARGTYPE_QUATERNION:
            type_funcs.quaternion_from_pyobject(value, <Quaternion *>arg_value_ptr)
        elif arg_type == ARGTYPE_AABB:
            type_funcs.aabb_from_pyobject(value, <_AABB *>arg_value_ptr)
        elif arg_type == ARGTYPE_BASIS:
            type_funcs.basis_from_pyobject(value, <Basis *>arg_value_ptr)
        elif arg_type == ARGTYPE_TRANSFORM3D:
            type_funcs.transform3d_from_pyobject(value, <Transform3D *>arg_value_ptr)
        elif arg_type == ARGTYPE_PROJECTION:
            type_funcs.projection_from_pyobject(value, <Projection *>arg_value_ptr)
        elif arg_type == ARGTYPE_COLOR:
            type_funcs.color_from_pyobject(value, <Color *>arg_value_ptr)
        elif arg_type == ARGTYPE_STRING_NAME:
            type_funcs.string_name_from_pyobject(value, <StringName *>arg_value_ptr)
        elif arg_type == ARGTYPE_NODE_PATH:
            type_funcs.node_path_from_pyobject(value, <NodePath *>arg_value_ptr)
        elif arg_type == ARGTYPE_RID:
            type_funcs.rid_from_pyobject(value, <_RID *>arg_value_ptr)
        elif arg_type == ARGTYPE_OBJECT:
            object_from_pyobject(value, <void **>arg_value_ptr)
        elif arg_type == ARGTYPE_CALLABLE:
            type_funcs.callable_from_pyobject(value, <GodotCppCallable *>arg_value_ptr)
        elif arg_type == ARGTYPE_SIGNAL:
            type_funcs.signal_from_pyobject(value, <GodotCppSignal *>arg_value_ptr)
        elif arg_type == ARGTYPE_DICTIONARY:
            type_funcs.dictionary_from_pyobject(value, <Dictionary *>arg_value_ptr)
        elif arg_type == ARGTYPE_ARRAY:
            type_funcs.array_from_pyobject(value, <Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_BYTE_ARRAY:
            type_funcs.packed_byte_array_from_pyobject(value, <PackedByteArray *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_INT32_ARRAY:
            type_funcs.packed_int32_array_from_pyobject(value, <PackedInt32Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_INT64_ARRAY:
            type_funcs.packed_int64_array_from_pyobject(value, <PackedInt64Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
            type_funcs.packed_float32_array_from_pyobject(value, <PackedFloat32Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
            type_funcs.packed_float64_array_from_pyobject(value, <PackedFloat64Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_STRING_ARRAY:
            type_funcs.packed_string_array_from_pyobject(value, <PackedStringArray *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
            type_funcs.packed_vector2_array_from_pyobject(value, <PackedVector2Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
            type_funcs.packed_vector3_array_from_pyobject(value, <PackedVector3Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_COLOR_ARRAY:
            type_funcs.packed_color_array_from_pyobject(value, <PackedColorArray *>arg_value_ptr)
        elif arg_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
            type_funcs.packed_vector4_array_from_pyobject(value, <PackedVector4Array *>arg_value_ptr)
        elif arg_type == ARGTYPE_VARIANT:
            type_funcs.variant_from_pyobject(value, <Variant *>arg_value_ptr)
        elif arg_type == ARGTYPE_POINTER:
            # Special case: ObjectID pointers are passed as void*
            if type(value) is type_funcs.ObjectID:
                type_funcs.object_id_from_pyobject(value, <ObjectID *>arg_value_ptr)
            else:
                type_funcs.pointer_from_pyobject(value, &arg_value_ptr)
        elif arg_type == ARGTYPE_AUDIO_FRAME:
            type_funcs.audio_frame_from_pyobject(value, <AudioFrame *>arg_value_ptr)
        elif arg_type == ARGTYPE_CARET_INFO:
            type_funcs.caret_info_from_pyobject(value, <CaretInfo *>arg_value_ptr)
        elif arg_type == ARGTYPE_GLYPH:
            type_funcs.glyph_from_pyobject(value, <Glyph *>arg_value_ptr)
        elif arg_type == ARGTYPE_OBJECT_ID:
            type_funcs.object_id_from_pyobject(value, <ObjectID *>arg_value_ptr)
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_MOTION_RESULT:
            type_funcs.physics_server2d_extension_motion_result_from_pyobject(
                value,
                <PhysicsServer2DExtensionMotionResult *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_RAY_RESULT:
            type_funcs.physics_server2d_extension_ray_result_from_pyobject(
                value,
                <PhysicsServer2DExtensionRayResult *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_REST_INFO:
            type_funcs.physics_server2d_extension_shape_rest_info_from_pyobject(
                value,
                <PhysicsServer2DExtensionShapeRestInfo *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_RESULT:
            type_funcs.physics_server2d_extension_shape_result_from_pyobject(
                value,
                <PhysicsServer2DExtensionShapeResult *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_COLLISION:
            type_funcs.physics_server3d_extension_motion_collision_from_pyobject(
                value,
                <PhysicsServer3DExtensionMotionCollision *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_RESULT:
            type_funcs.physics_server3d_extension_motion_result_from_pyobject(
                value,
                <PhysicsServer3DExtensionMotionResult *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_RAY_RESULT:
            type_funcs.physics_server3d_extension_ray_result_from_pyobject(
                value,
                <PhysicsServer3DExtensionRayResult *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_REST_INFO:
            type_funcs.physics_server3d_extension_shape_rest_info_from_pyobject(
                value,
                <PhysicsServer3DExtensionShapeRestInfo *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_RESULT:
            type_funcs.physics_server3d_extension_shape_result_from_pyobject(
                value,
                <PhysicsServer3DExtensionShapeResult *>arg_value_ptr
            )
        elif arg_type == ARGTYPE_SCRIPTING_LANGUAGE_PROFILING_INFO:
            type_funcs.script_language_extension_profiling_info_from_pyobject(
                value,
                <ScriptLanguageExtensionProfilingInfo *>arg_value_ptr
            )
        else:
            msg = "Could not convert argument '%s[#%d]' from %r in %r" \
                  % (method.type_info[i + 1], arg_type, value, method)
            UtilityFunctions.printerr(msg)

            raise GDExtensionEnginePtrCallError(msg)

        (<GDExtensionUninitializedTypePtr *>args_mem.ptr)[i] = arg_value_ptr


    cdef int8_t return_type = type_info[0]
    cdef void *ret_value_ptr

    if max_size > 0:
        ret_value_ptr = numpy.PyArray_GETPTR1(arg_values, 0)

    if return_type == ARGTYPE_NIL:
        ptrcall(method, NULL, <const void **>args_mem.ptr, size)
    else:
        if max_size == 0:
            raise GDExtensionEnginePtrCallError("Attempt to return a value of zero size")

        ptrcall(method, ret_value_ptr, <const void **>args_mem.ptr, size)

    # NOTE: Cython compiles this to C switch/case
    if return_type == ARGTYPE_NIL:
        value = None
    elif return_type == ARGTYPE_BOOL:
        value = type_funcs.bool_to_pyobject(deref(<uint8_t *>ret_value_ptr))
    elif return_type == ARGTYPE_INT:
        value = type_funcs.int_to_pyobject(deref(<int64_t *>ret_value_ptr))
    elif return_type == ARGTYPE_FLOAT:
        value = type_funcs.float_to_pyobject(deref(<double *>ret_value_ptr))
    elif return_type == ARGTYPE_STRING:
        value = type_funcs.string_to_pyobject(deref(<String *>ret_value_ptr))
    elif return_type == ARGTYPE_VECTOR2:
        value = type_funcs.vector2_to_pyobject(deref(<Vector2 *>ret_value_ptr))
    elif return_type == ARGTYPE_VECTOR2I:
        value = type_funcs.vector2i_to_pyobject(deref(<Vector2i *>ret_value_ptr))
    elif return_type == ARGTYPE_RECT2:
        value = type_funcs.rect2_to_pyobject(deref(<Rect2 *>ret_value_ptr))
    elif return_type == ARGTYPE_RECT2I:
        value = type_funcs.rect2i_to_pyobject(deref(<Rect2i *>ret_value_ptr))
    elif return_type == ARGTYPE_VECTOR3:
        value = type_funcs.vector3_to_pyobject(deref(<Vector3 *>ret_value_ptr))
    elif return_type == ARGTYPE_VECTOR3I:
        value = type_funcs.vector3i_to_pyobject(deref(<Vector3i *>ret_value_ptr))
    elif return_type == ARGTYPE_TRANSFORM2D:
        value = type_funcs.transform2d_to_pyobject(deref(<Transform2D *>ret_value_ptr))
    elif return_type == ARGTYPE_VECTOR4:
        value = type_funcs.vector4_to_pyobject(deref(<Vector4 *>ret_value_ptr))
    elif return_type == ARGTYPE_VECTOR4I:
        value = type_funcs.vector4i_to_pyobject(deref(<Vector4i *>ret_value_ptr))
    elif return_type == ARGTYPE_PLANE:
        value = type_funcs.plane_to_pyobject(deref(<Plane *>ret_value_ptr))
    elif return_type == ARGTYPE_QUATERNION:
        value = type_funcs.quaternion_to_pyobject(deref(<Quaternion *>ret_value_ptr))
    elif return_type == ARGTYPE_AABB:
        value = type_funcs.aabb_to_pyobject(deref(<_AABB *>ret_value_ptr))
    elif return_type == ARGTYPE_BASIS:
        value = type_funcs.basis_to_pyobject(deref(<Basis *>ret_value_ptr))
    elif return_type == ARGTYPE_TRANSFORM3D:
        value = type_funcs.transform3d_to_pyobject(deref(<Transform3D *>ret_value_ptr))
    elif return_type == ARGTYPE_PROJECTION:
        value = type_funcs.projection_to_pyobject(deref(<Projection *>ret_value_ptr))
    elif return_type == ARGTYPE_COLOR:
        value = type_funcs.color_to_pyobject(deref(<Color *>ret_value_ptr))
    elif return_type == ARGTYPE_STRING_NAME:
        value = type_funcs.string_name_to_pyobject(deref(<StringName *>ret_value_ptr))
    elif return_type == ARGTYPE_NODE_PATH:
        value = type_funcs.node_path_to_pyobject(deref(<NodePath *>ret_value_ptr))
    elif return_type == ARGTYPE_RID:
        value = type_funcs.rid_to_pyobject(deref(<_RID *>ret_value_ptr))
    elif return_type == ARGTYPE_OBJECT:
        value = object_to_pyobject(deref(<void **>ret_value_ptr))
    elif return_type == ARGTYPE_CALLABLE:
        value = type_funcs.callable_to_pyobject(deref(<GodotCppCallable *>ret_value_ptr))
    elif return_type == ARGTYPE_SIGNAL:
        value = type_funcs.signal_to_pyobject(deref(<GodotCppSignal *>ret_value_ptr))
    elif return_type == ARGTYPE_DICTIONARY:
        value = type_funcs.dictionary_to_pyobject(deref(<Dictionary *>ret_value_ptr))
    elif return_type == ARGTYPE_ARRAY:
        value = type_funcs.array_to_pyobject(deref(<Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_BYTE_ARRAY:
        value = type_funcs.packed_byte_array_to_pyobject(deref(<PackedByteArray *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_INT32_ARRAY:
        value = type_funcs.packed_int32_array_to_pyobject(deref(<PackedInt32Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_INT64_ARRAY:
        value = type_funcs.packed_int64_array_to_pyobject(deref(<PackedInt64Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_FLOAT32_ARRAY:
        value = type_funcs.packed_float32_array_to_pyobject(deref(<PackedFloat32Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_FLOAT64_ARRAY:
        value = type_funcs.packed_float64_array_to_pyobject(deref(<PackedFloat64Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_STRING_ARRAY:
        value = type_funcs.packed_string_array_to_pyobject(deref(<PackedStringArray *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_VECTOR2_ARRAY:
        value = type_funcs.packed_vector2_array_to_pyobject(deref(<PackedVector2Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_VECTOR3_ARRAY:
        value = type_funcs.packed_vector3_array_to_pyobject(deref(<PackedVector3Array *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_COLOR_ARRAY:
        value = type_funcs.packed_color_array_to_pyobject(deref(<PackedColorArray *>ret_value_ptr))
    elif return_type == ARGTYPE_PACKED_VECTOR4_ARRAY:
        value = type_funcs.packed_vector4_array_to_pyobject(deref(<PackedVector4Array *>ret_value_ptr))
    elif return_type == ARGTYPE_VARIANT:
        value = type_funcs.variant_to_pyobject(deref(<Variant *>ret_value_ptr))
    elif return_type == ARGTYPE_POINTER:
        value = type_funcs.pointer_to_pyobject(deref(<void **>ret_value_ptr))
    elif return_type == ARGTYPE_AUDIO_FRAME:
        value = type_funcs.audio_frame_to_pyobject(deref(<AudioFrame **>ret_value_ptr))
    elif return_type == ARGTYPE_CARET_INFO:
        value = type_funcs.caret_info_to_pyobject(deref(<CaretInfo **>ret_value_ptr))
    elif return_type == ARGTYPE_GLYPH:
        value = type_funcs.glyph_to_pyobject(deref(<Glyph **>ret_value_ptr))
    elif return_type == ARGTYPE_OBJECT_ID:
        value = type_funcs.object_id_to_pyobject(deref(<ObjectID **>ret_value_ptr))
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_MOTION_RESULT:
        value = type_funcs.physics_server2d_extension_motion_result_to_pyobject(
            deref(<PhysicsServer2DExtensionMotionResult **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_RAY_RESULT:
        value = type_funcs.physics_server2d_extension_ray_result_to_pyobject(
            deref(<PhysicsServer2DExtensionRayResult **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_REST_INFO:
        value = type_funcs.physics_server2d_extension_shape_rest_info_to_pyobject(
            deref(<PhysicsServer2DExtensionShapeRestInfo **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER2D_SHAPE_RESULT:
        value = type_funcs.physics_server2d_extension_shape_result_to_pyobject(
            deref(<PhysicsServer2DExtensionShapeResult **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_COLLISION:
        value = type_funcs.physics_server3d_extension_motion_collision_to_pyobject(
            deref(<PhysicsServer3DExtensionMotionCollision **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_MOTION_RESULT:
        value = type_funcs.physics_server3d_extension_motion_result_to_pyobject(
            deref(<PhysicsServer3DExtensionMotionResult **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_RAY_RESULT:
        value = type_funcs.physics_server3d_extension_ray_result_to_pyobject(
            deref(<PhysicsServer3DExtensionRayResult **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_REST_INFO:
        value = type_funcs.physics_server3d_extension_shape_rest_info_to_pyobject(
            deref(<PhysicsServer3DExtensionShapeRestInfo **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_PHYSICS_SERVER3D_SHAPE_RESULT:
        value = type_funcs.physics_server3d_extension_shape_result_to_pyobject(
            deref(<PhysicsServer3DExtensionShapeResult **>ret_value_ptr)
        )
    elif return_type == ARGTYPE_SCRIPTING_LANGUAGE_PROFILING_INFO:
        value = type_funcs.script_language_extension_profiling_info_to_pyobject(
            deref(<const ScriptLanguageExtensionProfilingInfo **>ret_value_ptr)
        )
    else:
        msg = "Could not convert return value '%s[#%d]' in %r" % (method.type_info[0], return_type, method)
        raise GDExtensionEnginePtrCallError(msg)

    return value
