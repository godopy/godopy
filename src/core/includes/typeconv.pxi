# cdef GDExtensionTypeFromVariantConstructorFunc to_type_constructor[<int>VARIANT_MAX]
# cdef GDExtensionVariantFromTypeConstructorFunc from_type_constructor[<int>VARIANT_MAX]
# cdef size_t i

# for i in range(1, VARIANT_MAX):
#     to_type_constructor[i] = gdextension_interface_get_variant_to_type_constructor(<GDExtensionVariantType>i)
#     from_type_constructor[i] = gdextension_interface_get_variant_from_type_constructor(<GDExtensionVariantType>i)


cdef dict TYPEMAP = {
    NIL: 'Nil',

    # atomic types
    BOOL: 'bool',
    INT: 'int',
    FLOAT: 'float',
    STRING: 'String',

    # math types
    VECTOR2: 'Vector2',
    VECTOR2I: 'Vector2i',
    RECT2: 'Rect2',
    RECT2I: 'Rect2i',
    VECTOR3: 'Vector3',
    VECTOR3I: 'Vector3i',
    TRANSFORM2D: 'Transform2D',
    VECTOR4: 'Vector4',
    VECTOR4I: 'Vector4i',
    PLANE: 'Plane',
    QUATERNION: 'Quaternion',
    AABB: 'AABB',
    BASIS: 'Basis',
    TRANSFORM3D: 'Transform3D',
    PROJECTION: 'Projection',

    # misc types
    COLOR: 'Color',
    STRING_NAME: 'StringName',
    NODE_PATH: 'NodePath',
    RID: 'RID',
    OBJECT: 'Object',
    CALLABLE: 'Callable',
    SIGNAL: 'Signal',
    DICTIONARY: 'Dictionary',
    ARRAY: 'Array',

    # typed arrays
    PACKED_BYTE_ARRAY: 'PackedByteArray',
    PACKED_INT32_ARRAY: 'PackedInt32Array',
    PACKED_INT64_ARRAY: 'PackedInt64Array',
    PACKED_FLOAT32_ARRAY: 'PackedFloat32Array',
    PACKED_FLOAT64_ARRAY: 'PackedFloat64Array',
    PACKED_STRING_ARRAY: 'PackedStringArray',
    PACKED_VECTOR2_ARRAY: 'PackedVector2Array',
    PACKED_VECTOR3_ARRAY: 'PackedVector3Array',
    PACKED_COLOR_ARRAY: 'PackedColorArray',
    PACKED_VECTOR4_ARRAY: 'PackedVector4Array',

    VARIANT_MAX: 'Variant'
}


cdef dict TYPEMAP_REVERSED = {v: k for k, v in TYPEMAP.items()}


cpdef str variant_type_to_str(VariantType vartype):
    return TYPEMAP[vartype]


cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX:
    return TYPEMAP_REVERSED[vartype]


cdef int make_optimized_type_info(object type_info, int8_t *opt_type_info) except -1:
    cdef size_t size = len(type_info), i
    cdef int8_t arg_info = -1
    cdef str arg_type

    for i in range(size):
        arg_type = type_info[i]
        arg_info = <int8_t>TYPEMAP_REVERSED.get(arg_type, -1)
        if arg_info >= 0:
            opt_type_info[i] = arg_info
            continue
        elif arg_type.startswith('enum:') or arg_type.startswith('bitfield:'):
            opt_type_info[i] = <int8_t>INT
        elif arg_type.startswith('typedarray:'):
            opt_type_info[i] = <int8_t>ARRAY
        elif arg_type.startswith('typeddict'):
            opt_type_info[i] = <int8_t>DICTIONARY
        elif arg_type in _global_inheritance_info:
            opt_type_info[i] = <int8_t>OBJECT
        else:
            UtilityFunction.push_error("NOT IMPLEMENTED: Could not detect argument type %r" % arg_type)
            # Possible types:
            #   builtin structs (CaretInfo, Glyph, etc)
            #   pointers to builtin structs (AudioFrame*)
            #   int/float/void pointers (various data buffers)
            # They need special handling
            # TODO: Support all extra types
            opt_type_info[i] = -1
