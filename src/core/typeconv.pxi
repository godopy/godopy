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


cpdef VariantType pytype_to_gdtype(object pytype):
    if pytype is None:
        return NIL

    # Atomic types
    elif issubclass(pytype, bool):
        return BOOL
    elif issubclass(pytype, int):
        return INT
    elif issubclass(pytype, float):
        return FLOAT
    elif issubclass(pytype, (str, bytes)):
        return STRING

    # Math types
    elif issubclass(pytype, gdtypes.Vector2):
        return VECTOR2
    elif issubclass(pytype, gdtypes.Vector2i):
        return VECTOR2I
    elif issubclass(pytype, gdtypes.Rect2):
        return RECT2
    elif issubclass(pytype, gdtypes.Rect2i):
        return RECT2I
    elif issubclass(pytype, gdtypes.Vector3):
        return VECTOR3
    elif issubclass(pytype, gdtypes.Vector3i):
        return VECTOR3I
    elif issubclass(pytype, gdtypes.Transform2D):
        return TRANSFORM2D
    elif issubclass(pytype, gdtypes.Vector4):
        return VECTOR4
    elif issubclass(pytype, gdtypes.Vector4i):
        return VECTOR4I
    elif issubclass(pytype, gdtypes.Plane):
        return PLANE
    elif issubclass(pytype, gdtypes.Quaternion):
        return QUATERNION
    elif issubclass(pytype, gdtypes.AABB):
        return AABB
    elif issubclass(pytype, gdtypes.Basis):
        return BASIS
    elif issubclass(pytype, gdtypes.Transform3D):
        return TRANSFORM3D
    elif issubclass(pytype, gdtypes.Projection):
        return PROJECTION

    # Misc types
    elif issubclass(pytype, gdtypes.Color):
        return COLOR
    elif issubclass(pytype, gdtypes.StringName):
        return STRING_NAME
    elif issubclass(pytype, gdtypes.NodePath):
        return NODE_PATH
    elif issubclass(pytype, gdtypes.RID):
        return RID
    elif issubclass(pytype, Callable):
        return CALLABLE
    elif issubclass(pytype, Signal):
        return SIGNAL
    elif issubclass(pytype, Object) or pytype is object:
        return OBJECT
    elif issubclass(pytype, dict):
        return DICTIONARY
    elif issubclass(pytype, list):
        return ARRAY

    # Typed arrays
    elif issubclass(pytype, bytearray):
        return PACKED_BYTE_ARRAY
    elif issubclass(pytype, gdtypes.PackedInt32Array):
        return PACKED_INT32_ARRAY
    elif issubclass(pytype, gdtypes.PackedInt64Array):
        return PACKED_INT64_ARRAY
    elif issubclass(pytype, gdtypes.PackedFloat32Array):
        return PACKED_FLOAT32_ARRAY
    elif issubclass(pytype, gdtypes.PackedFloat642Array):
        return PACKED_FLOAT64_ARRAY
    elif issubclass(pytype, gdtypes.PackedStringArray) or pytype is tuple:
        return PACKED_STRING_ARRAY
    elif issubclass(pytype, gdtypes.PackedVector2Array):
        return PACKED_VECTOR2_ARRAY
    elif issubclass(pytype, gdtypes.PackedVector3Array):
        return PACKED_VECTOR3_ARRAY
    elif issubclass(pytype, gdtypes.PackedColorArray):
        return PACKED_COLOR_ARRAY
    elif issubclass(pytype, gdtypes.PackedVector4Array):
        return PACKED_VECTOR4_ARRAY

    raise TypeError("No suitable conversion from %r to Godot Variant type found" % pytype)
