from libc.stddef cimport wchar_t
# from _gdextension_interface cimport *

cdef extern from "godot_cpp/variant/variant.hpp" namespace "godot" nogil:
    cdef cppclass String
    cdef cppclass StringName
    cdef cppclass Array
    cdef cppclass Dictionary

    cdef cppclass Variant:
        Variant()
        Variant(bint)
        Variant(signed int)
        Variant(unsigned int)
        Variant(signed short)
        Variant(unsigned short)
        Variant(signed char)
        Variant(unsigned char)
        Variant(double)
        Variant(const String &)
        Variant(const StringName &)
        Variant(const Array &)
        Vartiant(const Dictionary &)
        Variant(const char *)
        Variant(const wchar_t *)
        Variant(object)

        void *_native_ptr()
        object pythonize() const
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
