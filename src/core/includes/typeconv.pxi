# cdef GDExtensionTypeFromVariantConstructorFunc to_type_constructor[<int>VARIANT_MAX]
# cdef GDExtensionVariantFromTypeConstructorFunc from_type_constructor[<int>VARIANT_MAX]
# cdef size_t i

# for i in range(1, VARIANT_MAX):
#     to_type_constructor[i] = gdextension_interface_get_variant_to_type_constructor(<GDExtensionVariantType>i)
#     from_type_constructor[i] = gdextension_interface_get_variant_from_type_constructor(<GDExtensionVariantType>i)


cdef list TYPE_LIST = [
    'Nil',

    # atomic types
    'bool',
    'int',
    'float',
    'String',

    # math types
    'Vector2',
    'Vector2i',
    'Rect2',
    'Rect2i',
    'Vector3',
    'Vector3i',
    'Transform2D',
    'Vector4',
    'Vector4i',
    'Plane',
    'Quaternion',
    'AABB',
    'Basis',
    'Transform3D',
    'Projection',

    # misc types
    'Color',
    'StringName',
    'NodePath',
    'RID',
    'Object',
    'Callable',
    'Signal',
    'Dictionary',
    'Array',

    # typed arrays
    'PackedByteArray',
    'PackedInt32Array',
    'PackedInt64Array',
    'PackedFloat32Array',
    'PackedFloat64Array',
    'PackedStringArray',
    'PackedVector2Array',
    'PackedVector3Array',
    'PackedColorArray',
    'PackedVector4Array',
]

cdef list TYPE_SIZE_LIST = [
    0,

    cython.sizeof(uint8_t),
    cython.sizeof(int64_t),
    cython.sizeof(double),
    cython.sizeof(String),

    cython.sizeof(Vector2),
    cython.sizeof(Vector2i),
    cython.sizeof(Rect2),
    cython.sizeof(Rect2i),
    cython.sizeof(Vector3),
    cython.sizeof(Vector3i),
    cython.sizeof(Transform2D),
    cython.sizeof(Vector4),
    cython.sizeof(Vector4i),
    cython.sizeof(Plane),
    cython.sizeof(Quaternion),
    cython.sizeof(_AABB),
    cython.sizeof(Basis),
    cython.sizeof(Transform3D),
    cython.sizeof(Projection),

    cython.sizeof(Color),
    cython.sizeof(StringName),
    cython.sizeof(NodePath),
    cython.sizeof(_RID),
    cython.sizeof(GodotCppObject),
    cython.sizeof(GodotCppCallable),
    cython.sizeof(GodotCppSignal),
    cython.sizeof(Dictionary),
    cython.sizeof(Array),

    cython.sizeof(PackedByteArray),
    cython.sizeof(PackedInt32Array),
    cython.sizeof(PackedInt64Array),
    cython.sizeof(PackedFloat32Array),
    cython.sizeof(PackedFloat64Array),
    cython.sizeof(PackedStringArray),
    cython.sizeof(PackedVector2Array),
    cython.sizeof(PackedVector3Array),
    cython.sizeof(PackedColorArray),
    cython.sizeof(PackedVector4Array)
]

cdef list ARGTYPE_LIST = TYPE_LIST + [
    'Variant',
    'Pointer',
    'ScriptInstance',
    'AudioFrame',
    'CaretInfo',
    'Glyph',
    'ObjectID',
    'PhysicsServer2DExtensionMotionResult',
    'PhysicsServer2DExtensionRayResult',
    'PhysicsServer2DExtensionShapeRestInfo',
    'PhysicsServer2DExtensionShapeResult',
    'PhysicsServer3DExtensionMotionCollision',
    'PhysicsServer3DExtensionMotionResult',
    'PhysicsServer3DExtensionRayResult',
    'PhysicsServer3DExtensionShapeRestInfo',
    'PhysicsServer3DExtensionShapeResult',
    'ScriptLanguageExtensionProfilingInfo'
]

cdef size_t ptr_size = cython.sizeof(GDExtensionTypePtr)

cdef list ARGTYPE_SIZE_LIST = TYPE_SIZE_LIST + [
    cython.sizeof(Variant),
    ptr_size,
    ptr_size,

    # NOTE: All structs are passed by pointer
    cython.sizeof(AudioFrame),
    cython.sizeof(CaretInfo),
    cython.sizeof(Glyph),
    cython.sizeof(ObjectID),
    cython.sizeof(PhysicsServer2DExtensionMotionResult),
    cython.sizeof(PhysicsServer2DExtensionRayResult),
    cython.sizeof(PhysicsServer2DExtensionShapeRestInfo),
    cython.sizeof(PhysicsServer2DExtensionShapeResult),
    cython.sizeof(PhysicsServer3DExtensionMotionCollision),
    cython.sizeof(PhysicsServer3DExtensionMotionResult),
    cython.sizeof(PhysicsServer3DExtensionRayResult),
    cython.sizeof(PhysicsServer3DExtensionShapeRestInfo),
    cython.sizeof(PhysicsServer3DExtensionShapeResult),
    cython.sizeof(ScriptLanguageExtensionProfilingInfo)
]

cdef dict TYPE_MAP = {i: v for i, v in enumerate(TYPE_LIST)}
cdef dict TYPE_MAP_REVERSED = {v: i for i, v in enumerate(TYPE_LIST)}
cdef dict ARGTYPE_MAP = {i: v for i, v in enumerate(ARGTYPE_LIST)}
cdef dict ARGTYPE_MAP_REVERSED = {v: i for i, v in enumerate(ARGTYPE_LIST)}

cpdef str variant_type_to_str(VariantType vartype):
    return TYPE_MAP[vartype]


cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX:
    return TYPE_MAP_REVERSED[vartype]


cpdef str arg_type_to_str(ArgType argtype):
    return ARGTYPE_MAP[argtype]


cpdef ArgType str_to_arg_type(str argtype) except ARGTYPE_NO_ARGTYPE:
    return ARGTYPE_MAP_REVERSED[argtype]


cdef int get_max_arg_size(int8_t *opt_type_info, size_t arg_size) except -1:
    cdef int size = 0, i = 0
    for i in range(arg_size):
        size = max(size, <size_t>ARGTYPE_SIZE_LIST[opt_type_info[i]])

    return size


cdef int make_optimized_type_info(object type_info, int8_t *opt_type_info) except -1:
    cdef size_t size = len(type_info), i
    cdef int8_t arg_info = -1
    cdef str arg_type, arg_type_noptr

    for i in range(size):
        arg_type = type_info[i]
        if arg_type.startswith('const'):
            arg_type = arg_type[6:]

        arg_type_noptr = arg_type
        if arg_type.endswith('*'):
            arg_type_noptr = arg_type.rstrip('*').rstrip()
            if arg_type_noptr in _global_struct_info and not arg_type.endswith('*'):
                raise ArgumentError("Unexpected non-pointer structure argument %r" % arg_type)
        arg_info = <int8_t>ARGTYPE_MAP_REVERSED.get(arg_type_noptr, -1)
        if arg_info >= 0:
            opt_type_info[i] = arg_info
            continue
        elif arg_type.startswith('enum:') or arg_type.startswith('bitfield:'):
            opt_type_info[i] = <int8_t>ARGTYPE_INT
        elif arg_type.startswith('typedarray:'):
            opt_type_info[i] = <int8_t>ARGTYPE_ARRAY
        elif arg_type.startswith('typeddict'):
            opt_type_info[i] = <int8_t>ARGTYPE_DICTIONARY
        elif arg_type in _global_inheritance_info:
            opt_type_info[i] = <int8_t>ARGTYPE_OBJECT
        elif arg_type.endswith('*'):
            opt_type_info[i] = <int8_t>ARGTYPE_POINTER
        else:
            raise ArgumentError("Could not detect argument type %r" % arg_type)

            # opt_type_info[i] = ARGTYPE_NO_ARGTYPE
