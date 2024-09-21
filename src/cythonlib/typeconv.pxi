# cdef GDExtensionTypeFromVariantConstructorFunc to_type_constructor[<int>VARIANT_MAX]
# cdef GDExtensionVariantFromTypeConstructorFunc from_type_constructor[<int>VARIANT_MAX]
# cdef size_t i

# for i in range(1, VARIANT_MAX):
#     to_type_constructor[i] = _gde_get_variant_to_type_constructor(<GDExtensionVariantType>i)
#     from_type_constructor[i] = _gde_get_variant_from_type_constructor(<GDExtensionVariantType>i)


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
    # VariantType.RECT2I
    # VariantType.VECTOR3
    # VariantType.VECTOR3I
    # VariantType.TRANSFORM2D
    # VariantType.VECTOR4
    # VariantType.VECTOR4I
    # VariantType.PLANE
    # VariantType.QUATERNION
    # VariantType.AABB
    # VariantType.BASIS
    # VariantType.TRANSFORM3D
    # VariantType.PROJECTION

    # # misc types
    VariantType.COLOR: 'Color',
    VariantType.STRING_NAME: 'StringName',
    # VariantType.NODE_PATH
    # VariantType.RID
    # VariantType.OBJECT
    # VariantType.CALLABLE
    # VariantType.SIGNAL
    VariantType.DICTIONARY: 'Dictionary',
    VariantType.ARRAY: 'Array',

    # # typed arrays
    # VariantType.PACKED_BYTE_ARRAY
    # VariantType.PACKED_INT32_ARRAY
    # VariantType.PACKED_INT64_ARRAY
    # VariantType.PACKED_FLOAT32_ARRAY
    # VariantType.PACKED_FLOAT64_ARRAY
    # VariantType.PACKED_STRING_ARRAY
    # VariantType.PACKED_VECTOR2_ARRAY
    # VariantType.PACKED_VECTOR3_ARRAY
    # VariantType.PACKED_COLOR_ARRAY
    # VariantType.PACKED_VECTOR4_ARRAY

    # VariantType.VARIANT_MAX
}

cdef dict TYPEMAP_REVERSED = {v: k for k, v in TYPEMAP.iteritems()}

cpdef str variant_to_str(VariantType vartype):
    return TYPEMAP.get(vartype, vartype)

cpdef VariantType str_to_variant(str vartype):
    return TYPEMAP_REVERSED.get(vartype, -1)
