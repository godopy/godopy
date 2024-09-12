cdef inline StringName stringname_from_str(str s):
    cdef bytes b = s.encode('utf-8')
    cdef const char *cc = b
    return StringName(cc)


cdef inline str str_from_variant(Variant v):
    cdef String s = <String>v
    cdef CharString cs = s.utf8()
    cdef const char *data = cs.get_data()
    return data.decode('utf-8')


cdef inline object pyobject_from_variant(const Variant &v):
    cdef VariantType t = <VariantType>v.get_type()
    if t == VariantType.STRING:
        return str_from_variant(v)
    elif t == VariantType.INT:
        return <int64_t>v
    elif t == VariantType.FLOAT:
        return <double>v
    elif t == VariantType.BOOL:
        return builtins.bool(<int>v)
    elif t == VariantType.OBJECT:
        raise NotImplementedError()
        # return wrap_cpp(<Object *>v)

    return None

cdef inline Variant variant_from_pyobject(object o):
    cdef Variant v
    if isinstance(o, str):
        b = o.encode('utf-8')
        v = Variant(<const char *>b)
    elif isinstance(o, bytes):
        v = Variant(<const char *>o)
    elif isinstance(o, builtins.bool):
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
