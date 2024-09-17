cdef TYPEMAP = {
    VariantType.NIL: 'Nil',

    # atomic types
    VariantType.BOOL: 'Bool',
    VariantType.INT: 'Int',
    VariantType.FLOAT: 'Float',
    VariantType.STRING: 'String',

    # math types
    VariantType.VECTOR2: 'Vector2',
    VariantType.VECTOR2I: 'Vector2i',
    # VariantType.RECT2
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
    # VariantType.COLOR
    # VariantType.STRING_NAME
    # VariantType.NODE_PATH
    # VariantType.RID
    # VariantType.OBJECT
    # VariantType.CALLABLE
    # VariantType.SIGNAL
    # VariantType.DICTIONARY
    # VariantType.ARRAY

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

def vartype_to_str(int vartype):
    return TYPEMAP.get(vartype, vartype)
