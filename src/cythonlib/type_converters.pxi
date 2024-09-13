cdef inline StringName stringname_from_str(str s):
    return StringName(s)


cdef inline str str_from_variant(Variant v):
    cdef String s = v.stringify()
    return s.py_str()


cdef inline object pyobject_from_variant(const Variant &v):
    return <object>v;


cdef inline Variant variant_from_pyobject(object o):
    return <Variant>o;
