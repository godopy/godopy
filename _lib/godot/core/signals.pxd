from godot_headers.gdnative_api cimport godot_int, godot_property_hint, godot_property_usage_flags

cdef class SignalArgument:
    cdef public str name
    cdef public godot_int type
    cdef public godot_property_hint hint
    cdef public str hint_string
    cdef public godot_property_usage_flags usage
    cdef public object default_value

cdef class SignalArgumentNil(SignalArgument): pass

cdef class SignalArgumentBool(SignalArgument): pass
cdef class SignalArgumentInt(SignalArgument): pass
cdef class SignalArgumentReal(SignalArgument): pass
cdef class SignalArgumentString(SignalArgument): pass

cdef class SignalArgumentVector2(SignalArgument): pass
cdef class SignalArgumentRect2(SignalArgument): pass
cdef class SignalArgumentVector3(SignalArgument): pass
cdef class SignalArgumentTransform2D(SignalArgument): pass
cdef class SignalArgumentPlane(SignalArgument): pass
cdef class SignalArgumentQuat(SignalArgument): pass
cdef class SignalArgumentRect3(SignalArgument): pass
cdef class SignalArgumentBasis(SignalArgument): pass
cdef class SignalArgumentTransform(SignalArgument): pass

cdef class SignalArgumentColor(SignalArgument): pass
cdef class SignalArgumentNodePath(SignalArgument): pass
cdef class SignalArgumentRID(SignalArgument): pass
cdef class SignalArgumentObject(SignalArgument): pass
cdef class SignalArgumentDictionary(SignalArgument): pass
cdef class SignalArgumentArray(SignalArgument): pass

cdef class SignalArgumentPoolByteArray(SignalArgument): pass
cdef class SignalArgumentPoolIntArray(SignalArgument): pass
cdef class SignalArgumentPoolRealArray(SignalArgument): pass
cdef class SignalArgumentPoolStringArray(SignalArgument): pass
cdef class SignalArgumentPoolVector2Array(SignalArgument): pass
cdef class SignalArgumentPoolVector3Array(SignalArgument): pass
cdef class SignalArgumentPoolColorArray(SignalArgument): pass
