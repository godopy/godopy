Nil = None


bool = bool


cdef public object bool_to_pyobject(GDExtensionBool p_bool):
    return p_bool != 0


cdef public object variant_bool_to_pyobject(const cpp.Variant &v):
    cdef bint ret = v.to_type[bint]()

    return ret != 0


# TODO: Keep type checks only in debug builds

cdef public void bool_from_pyobject(object p_obj, GDExtensionBool *r_ret) noexcept:
    if not PyBool_Check(p_obj):
        cpp.UtilityFunctions.push_error("'bool' is required, got %r" % type(p_obj))
        r_ret[0] = False
    else:
        r_ret[0] = <GDExtensionBool>PyObject_IsTrue(p_obj)


cdef public void *variant_bool_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef bint ret = False
    if not PyBool_Check(p_obj):
        cpp.UtilityFunctions.push_error("'bool' is required, got %r" % type(p_obj))
    else:
        ret = <GDExtensionBool>PyObject_IsTrue(p_obj)

    r_ret[0] = cpp.Variant(ret)


int = int


cdef public object int_to_pyobject(int64_t p_int):
    return p_int


cdef public object variant_int_to_pyobject(const cpp.Variant &v):
    cdef int64_t ret = v.to_type[int64_t]()

    return ret


cdef public void int_from_pyobject(object p_obj, int64_t *r_ret) noexcept:
    if not PyLong_Check(p_obj):
        cpp.UtilityFunctions.push_error("'int' is required, got %r" % type(p_obj))
        r_ret[0] = 0
    else:
        r_ret[0] = <int64_t>p_obj


cdef public void variant_int_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int64_t ret = 0

    if not PyLong_Check(p_obj):
        cpp.UtilityFunctions.push_error("'int' is required, got %r" % type(p_obj))
    else:
        ret = <int64_t>p_obj

    r_ret[0] = cpp.Variant(ret)


float = float


cdef public object float_to_pyobject(double p_float):
    return p_float


cdef public object variant_float_to_pyobject(const cpp.Variant &v):
    cdef double ret = v.to_type[double]()

    return ret


cdef public void float_from_pyobject(object p_obj, double *r_ret) noexcept:
    if not PyFloat_Check(p_obj):
        cpp.UtilityFunctions.push_error("'float' is required, got %r" % p_obj)
        r_ret[0] = 0.0
    else:
        r_ret[0] = <double>p_obj


cdef public void variant_float_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef double ret = 0.0

    if not PyFloat_Check(p_obj):
        cpp.UtilityFunctions.push_error("'float' is required, got %r" % p_obj)
    else:
        ret = <double>p_obj

    r_ret[0] = cpp.Variant(ret)


String = str


cdef public object string_to_pyobject(const cpp.String &p_string):
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(p_string._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(p_string._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return PyUnicode_FromWideChar(wstr.get_data(), len)


cdef public object variant_string_to_pyobject(const cpp.Variant &v):
    cdef cpp.String ret = v.to_type[cpp.String]()
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(ret._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(ret._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return PyUnicode_FromWideChar(wstr.get_data(), len)


cdef public void string_from_pyobject(object p_obj, cpp.String *r_ret) noexcept:
    cdef const wchar_t *wstr
    cdef const char *cstr

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars(r_ret, wstr)
    elif PyBytes_Check(p_obj):
        cstr = <bytes>p_obj
        gdextension_interface_string_new_with_utf8_chars(r_ret, cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ String" % p_obj)
        r_ret[0] = cpp.String()


cdef public void variant_string_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef const wchar_t * wstr
    cdef const char * cstr

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        r_ret[0] = cpp.Variant(wstr)
    elif PyBytes_Check(p_obj):
        cstr = <bytes>p_obj
        r_ret[0] = cpp.Variant(cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ String" % p_obj)
        cstr = ""
        r_ret[0] = cpp.Variant(cstr)
