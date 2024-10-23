# cdef GDExtensionTypeFromVariantConstructorFunc to_type_constructor[<int>VARIANT_MAX]
# cdef GDExtensionVariantFromTypeConstructorFunc from_type_constructor[<int>VARIANT_MAX]
# cdef size_t i

# for i in range(1, VARIANT_MAX):
#     to_type_constructor[i] = gdextension_interface_get_variant_to_type_constructor(<GDExtensionVariantType>i)
#     from_type_constructor[i] = gdextension_interface_get_variant_from_type_constructor(<GDExtensionVariantType>i)


cdef dict TYPEMAP = {
    VariantType.NIL: 'Nil',

    # atomic types
    VariantType.BOOL: 'bool',
    VariantType.INT: 'int',
    VariantType.FLOAT: 'float',
    VariantType.STRING: 'String',

    # math types
    VariantType.VECTOR2: 'Vector2',
    VariantType.VECTOR2I: 'Vector2i',
    VariantType.RECT2: 'Rect2',
    VariantType.RECT2I: 'Rect2i',
    VariantType.VECTOR3: 'Vector3',
    VariantType.VECTOR3I: 'Vector3i',
    VariantType.TRANSFORM2D: 'Transform2D',
    VariantType.VECTOR4: 'Vector4',
    VariantType.VECTOR4I: 'Vector4i',
    VariantType.PLANE: 'Plane',
    VariantType.QUATERNION: 'Quaternion',
    VariantType.AABB: 'AABB',
    VariantType.BASIS: 'Basis',
    VariantType.TRANSFORM3D: 'Transform3D',
    VariantType.PROJECTION: 'Projection',

    # misc types
    VariantType.COLOR: 'Color',
    VariantType.STRING_NAME: 'StringName',
    VariantType.NODE_PATH: 'NodePath',
    VariantType.RID: 'RID',
    VariantType.OBJECT: 'Object',
    VariantType.CALLABLE: 'Callable',
    VariantType.SIGNAL: 'Signal',
    VariantType.DICTIONARY: 'Dictionary',
    VariantType.ARRAY: 'Array',

    # typed arrays
    VariantType.PACKED_BYTE_ARRAY: 'PackedByteArray',
    VariantType.PACKED_INT32_ARRAY: 'PackedInt32Array',
    VariantType.PACKED_INT64_ARRAY: 'PackedInt64Array',
    VariantType.PACKED_FLOAT32_ARRAY: 'PackedFloat32Array',
    VariantType.PACKED_FLOAT64_ARRAY: 'PackedFloat64Array',
    VariantType.PACKED_STRING_ARRAY: 'PackedStringArray',
    VariantType.PACKED_VECTOR2_ARRAY: 'PackedVector2Array',
    VariantType.PACKED_VECTOR3_ARRAY: 'PackedVector3Array',
    VariantType.PACKED_COLOR_ARRAY: 'PackedColorArray',
    VariantType.PACKED_VECTOR4_ARRAY: 'PackedVector4Array'
}


cdef dict TYPEMAP_REVERSED = {v: k for k, v in TYPEMAP.items()}


cpdef str variant_type_to_str(VariantType vartype):
    return TYPEMAP.get(vartype, vartype)


cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX:
    return TYPEMAP_REVERSED[vartype]
