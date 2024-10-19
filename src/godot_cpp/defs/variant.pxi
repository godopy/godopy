from cpython cimport PyObject

cdef extern from "godot_cpp/variant/variant.hpp" namespace "godot" nogil:
    cdef cppclass String
    cdef cppclass Vector2
    cdef cppclass Vector2i
    cdef cppclass Rect2
    cdef cppclass Rect2i
    cdef cppclass Vector3
    cdef cppclass Vector3i
    cdef cppclass Transform2D
    cdef cppclass Vector4
    cdef cppclass Vector4i
    cdef cppclass Plane
    cdef cppclass Quaternion
    cdef cppclass _AABB
    cdef cppclass Basis
    cdef cppclass Transform3D
    cdef cppclass Projection
    cdef cppclass Color
    cdef cppclass StringName
    cdef cppclass NodePath
    cdef cppclass _RID
    cdef cppclass _Object
    cdef cppclass _Callable
    cdef cppclass _Signal
    cdef cppclass Dictionary
    cdef cppclass Array
    cdef cppclass PackedByteArray
    cdef cppclass PackedInt32Array
    cdef cppclass PackedInt64Array
    cdef cppclass PackedStringArray
    cdef cppclass PackedVector2Array
    cdef cppclass PackedVector3Array
    cdef cppclass PackedColorArray
    cdef cppclass PackedVector4Array

    cdef cppclass Variant:
        Variant()
        Variant(bint)
        Variant(GDExtensionBool)
        Variant(signed int)
        Variant(unsigned int)
        Variant(signed short)
        Variant(unsigned short)
        Variant(signed char)
        Variant(unsigned char)
        Variant(int8_t)
        Variant(int16_t)
        Variant(int32_t)
        Variant(int64_t)
        Variant(uint64_t)
        Variant(float)
        Variant(double)
        Variant(const String &)
        Variant(const Vector2 &)
        Variant(const Vector2i &)
        Variant(const Rect2 &)
        Variant(const Rect2i &)
        Variant(const Transform2D &)
        Variant(const Vector4 &)
        Variant(const Vector4i &)
        Variant(const Plane &)
        Variant(const Quaternion &)
        Variant(const _AABB &)
        Variant(const Basis &)
        Variant(const Transform3D &)
        Variant(const Projection &)
        Variant(const Color &)
        Variant(const StringName &)
        Variant(const NodePath &)
        Variant(const _RID &)
        Variant(void *)
        Variant(const _Object &)
        Variant(const _Callable &)
        Variant(const _Signal &)
        Variant(const Dictionary &)
        Variant(const Array &)
        Variant(const PackedByteArray &)
        Variant(const PackedInt32Array &)
        Variant(const PackedInt64Array &)
        Variant(const PackedFloat32Array &)
        Variant(const PackedFloat64Array &)
        Variant(const PackedStringArray &)
        Variant(const PackedVector2Array &)
        Variant(const PackedVector3Array &)
        Variant(const PackedColorArray &)
        Variant(const PackedVector4Array &)
        Variant(const PyObject *)

        T to_type[T]() const

        void *_native_ptr()
        object pythonize() const
        GDExtensionBool booleanize() const
        String stringify() const

        enum Type:
            NIL

            # atomic types
            BOOL
            INT
            FLOAT
            STRING

            # math types
            VECTOR2
            VECTOR2I
            RECT2
            RECT2I
            VECTOR3
            VECTOR3I
            TRANSFORM2D
            VECTOR4
            VECTOR4I
            PLANE
            QUATERNION
            AABB
            BASIS
            TRANSFORM3D
            PROJECTION

            # misc types
            COLOR
            STRING_NAME
            NODE_PATH
            RID
            OBJECT
            CALLABLE
            SIGNAL
            DICTIONARY
            ARRAY

            # typed arrays
            PACKED_BYTE_ARRAY
            PACKED_INT32_ARRAY
            PACKED_INT64_ARRAY
            PACKED_FLOAT32_ARRAY
            PACKED_FLOAT64_ARRAY
            PACKED_STRING_ARRAY
            PACKED_VECTOR2_ARRAY
            PACKED_VECTOR3_ARRAY
            PACKED_COLOR_ARRAY
            PACKED_VECTOR4_ARRAY

            VARIANT_MAX

        Type get_type()

cdef enum VariantType:
    NIL

    # atomic types
    BOOL
    INT
    FLOAT
    STRING

    # math types
    VECTOR2
    VECTOR2I
    RECT2
    RECT2I
    VECTOR3
    VECTOR3I
    TRANSFORM2D
    VECTOR4
    VECTOR4I
    PLANE
    QUATERNION
    AABB
    BASIS
    TRANSFORM3D
    PROJECTION

    # misc types
    COLOR
    STRING_NAME
    NODE_PATH
    RID
    OBJECT
    CALLABLE
    SIGNAL
    DICTIONARY
    ARRAY

    # typed arrays
    PACKED_BYTE_ARRAY
    PACKED_INT32_ARRAY
    PACKED_INT64_ARRAY
    PACKED_FLOAT32_ARRAY
    PACKED_FLOAT64_ARRAY
    PACKED_STRING_ARRAY
    PACKED_VECTOR2_ARRAY
    PACKED_VECTOR3_ARRAY
    PACKED_COLOR_ARRAY
    PACKED_VECTOR4_ARRAY

    VARIANT_MAX
