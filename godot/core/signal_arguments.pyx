from godot_headers.gdnative_api cimport GODOT_PROPERTY_HINT_NONE, GODOT_PROPERTY_USAGE_DEFAULT

from .defs cimport *

cdef class SignalArgument:
    def __init__(self, str name, godot_property_hint hint=GODOT_PROPERTY_HINT_NONE, str hint_string='',
                 godot_property_usage_flags usage=GODOT_PROPERTY_USAGE_DEFAULT, object default_value=None):
        self.name = name
        self.hint = hint
        self.hint_string = hint_string
        self.usage = usage
        self.default_value = default_value


cdef class SignalArgumentNil(SignalArgument):
    def __cinit__(self, str name):
        self.name = name
        self.type = <godot_int>VARIANT_NIL


cdef class SignalArgumentBool(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_BOOL

cdef class SignalArgumentInt(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_INT

cdef class SignalArgumentReal(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_REAL

cdef class SignalArgumentString(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_STRING


cdef class SignalArgumentVector2(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_VECTOR2

cdef class SignalArgumentRect2(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_RECT3

cdef class SignalArgumentVector3(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_VECTOR3

cdef class SignalArgumentTransform2D(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_TRANSFORM2D

cdef class SignalArgumentPlane(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_PLANE

cdef class SignalArgumentQuat(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_QUAT

cdef class SignalArgumentRect3(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_RECT3

cdef class SignalArgumentBasis(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_BASIS

cdef class SignalArgumentTransform(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_TRANSFORM


cdef class SignalArgumentColor(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_COLOR

cdef class SignalArgumentNodePath(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_NODE_PATH

cdef class SignalArgumentRID(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT__RID


cdef class SignalArgumentObject(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_OBJECT

    def __init__(self, str name, godot_property_hint hint=GODOT_PROPERTY_HINT_NONE, str hint_string='',
                 godot_property_usage_flags usage=GODOT_PROPERTY_USAGE_DEFAULT, object default_value=-1):
        self.name = name
        self.hint = hint
        self.hint_string = hint_string
        self.usage = usage
        self.default_value = default_value


cdef class SignalArgumentDictionary(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_DICTIONARY

cdef class SignalArgumentArray(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_ARRAY


cdef class SignalArgumentPoolByteArray(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_BYTE_ARRAY

cdef class SignalArgumentPoolIntArray(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_INT_ARRAY

cdef class SignalArgumentPoolRealArray(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_REAL_ARRAY

cdef class SignalArgumentPoolStringArray(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_STRING_ARRAY

cdef class SignalArgumentPoolVector2Array(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_VECTOR2_ARRAY

cdef class SignalArgumentPoolVector3Array(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_VECTOR3_ARRAY

cdef class SignalArgumentPoolColorArray(SignalArgument):
    def __cinit__(self, name):
        self.name = name
        self.type = <godot_int>VARIANT_POOL_COLOR_ARRAY
