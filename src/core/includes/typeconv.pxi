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

cdef list ARGTYPE_LIST = TYPE_LIST + [
    'Variant'
]

cdef dict TYPE_MAP = {i: v for i, v in enumerate(TYPE_LIST)}
cdef dict TYPE_MAP_REVERSED = {v: i for i, v in enumerate(TYPE_LIST)}
cdef dict ARGTYPE_MAP = {i: v for i, v in enumerate(ARGTYPE_LIST)}
cdef dict ARGTYPE_MAP_REVERSED = {v: i for i, v in enumerate(ARGTYPE_LIST)}

cpdef str variant_type_to_str(VariantType vartype):
    return TYPE_MAP[vartype]


cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX:
    return TYPE_MAP_REVERSED[vartype]


cpdef str arg_type_to_str(VariantType vartype):
    return ARGTYPE_MAP[vartype]


cpdef ArgType str_to_arg_type(str vartype) except ARGTYPE_MAX:
    return ARGTYPE_MAP_REVERSED[vartype]


cdef int make_optimized_type_info(object type_info, int8_t *opt_type_info) except -1:
    cdef size_t size = len(type_info), i
    cdef int8_t arg_info = -1
    cdef str arg_type

    for i in range(size):
        arg_type = type_info[i]
        arg_info = <int8_t>ARGTYPE_MAP_REVERSED.get(arg_type, -1)
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
