cdef inline StringName stringname_from_str(str s):
    return StringName(s)


cdef inline str str_from_variant(Variant v):
    cdef String s = <String>v
    return s.py_str()


cdef inline object pyobject_from_variant(const Variant &v):
    cdef VariantType t = <VariantType>v.get_type()
    if t == VariantType.STRING:
        return str_from_variant(v)
    elif t == VariantType.INT:
        return <int64_t>v
    elif t == VariantType.FLOAT:
        return <double>v
    elif t == VariantType.BOOL:
        return bool(<int>v)
    elif t == VariantType.OBJECT:
        raise NotImplementedError()
        # return wrap_cpp(<Object *>v)

    return None


cdef inline Variant variant_from_pyobject(object o):
    cdef Variant v
    if isinstance(o, str):
        v = Variant(<String>o)
    elif isinstance(o, bytes):
        v = Variant(<String>o)
    elif isinstance(o, bool):
        v = Variant(<bint>o)
    elif isinstance(o, int):
        v = Variant(<int64_t>o)
    elif isinstance(o, float):
        v = Variant(<double>o)
    elif isinstance(o, godot.GodotObject):
        raise NotImplementedError()
        # ref.Py_INCREF(o)
        # return <GDExtensionConstTypePtr>&Variant(<const Object *>(<GodotObject>o)._owner)
    else:
        v = Variant() # NIL

    return v
