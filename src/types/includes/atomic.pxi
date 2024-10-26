Nil = None


cdef public object typeptr_nil_to_pyobject(const void *v):
    return None


cdef public object variant_nil_to_pyobject(const cpp.Variant &v):
    return None


cdef public void typeptr_nil_from_pyobject(object p_obj, void *r_ret) noexcept:
    gdextension_interface_variant_new_nil(r_ret)


cdef public void variant_nil_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    # if p_obj is not None:
    #     cpp.UtilityFunctions.push_error("'None' is required, got %r" % type(p_obj))
    gdextension_interface_variant_new_nil(r_ret)


bool = bool


cdef public object bool_to_pyobject(const uint8_t p_bool):
    return p_bool != 0


cdef public object typeptr_bool_to_pyobject(const void *p_bool):
    return (<uint8_t *>p_bool)[0] != 0


cdef public object variant_bool_to_pyobject(const cpp.Variant &v):
    cdef uint8_t ret = v.to_type[bint]()

    return ret != 0


# TODO: Keep type checks only in debug builds

cdef public void bool_from_pyobject(object p_obj, uint8_t *r_ret) noexcept:
    # if not (PyBool_Check(p_obj) or isinstance(p_obj, np.bool_)):
    #     cpp.UtilityFunctions.push_error("'bool' is required, got %r" % type(p_obj))
    #     r_ret[0] = <uint8_t>False
    # else:
    r_ret[0] = <uint8_t>PyObject_IsTrue(p_obj)


cdef public void typeptr_bool_from_pyobject(object p_obj, void *r_ret) noexcept:
    (<uint8_t *>r_ret)[0] = <uint8_t>PyObject_IsTrue(p_obj)


cdef public void variant_bool_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef bint ret = False
    # if not (PyBool_Check(p_obj) or not isinstance(p_obj, np.bool_)):
    #     cpp.UtilityFunctions.push_error("'bool' is required, got %r" % type(p_obj))
    # else:
    ret = PyObject_IsTrue(p_obj)

    r_ret[0] = cpp.Variant(ret, True)


int = int


cdef public object int_to_pyobject(const int64_t p_int):
    return p_int


cdef public object typeptr_int_to_pyobject(const void *p_int):
    return (<int64_t *>p_int)[0]


cdef public object variant_int_to_pyobject(const cpp.Variant &v):
    cdef int64_t ret = v.to_type[int64_t]()

    return ret


cdef public void int_from_pyobject(object p_obj, int64_t *r_ret) noexcept:
    # if not (PyLong_Check(p_obj) or not isinstance(p_obj, np.integer)):
    #     cpp.UtilityFunctions.push_error("'int' is required, got %r" % type(p_obj))
    #     r_ret[0] = 0
    # else:

    r_ret[0] = <int64_t>p_obj


cdef public void typeptr_int_from_pyobject(object p_obj, void *r_ret) noexcept:
    (<int64_t *>r_ret)[0] = <int64_t>p_obj


cdef public void variant_int_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int64_t ret = 0

    # if not (PyLong_Check(p_obj) or not isinstance(p_obj, np.integer)):
    #     cpp.UtilityFunctions.push_error("'int' is required, got %r" % type(p_obj))
    # else:
    ret = <int64_t>p_obj

    r_ret[0] = cpp.Variant(ret)


float = float


cdef public object float_to_pyobject(const double p_float):
    return p_float


cdef public object typeptr_float_to_pyobject(const void *p_float):
    return (<double *>p_float)[0]


cdef public object variant_float_to_pyobject(const cpp.Variant &v):
    cdef double ret = v.to_type[double]()

    return ret


cdef public void float_from_pyobject(object p_obj, double *r_ret) noexcept:
    # if not (PyFloat_Check(p_obj) or not isinstance(p_obj, np.floating)):
    #     cpp.UtilityFunctions.push_error("'float' is required, got %r" % type(p_obj))
    #     r_ret[0] = 0.0
    # else:
    r_ret[0] = <double>p_obj


cdef public void typeptr_float_from_pyobject(object p_obj, void *r_ret) noexcept:
    (<double *>r_ret)[0] = <double>p_obj


cdef public void variant_float_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef double ret = 0.0

    if not (PyFloat_Check(p_obj) or not isinstance(p_obj, np.floating)):
        cpp.UtilityFunctions.push_error("'float' is required, got %r" % type(p_obj))
    else:
        ret = <double>p_obj

    r_ret[0] = cpp.Variant(ret)


def as_string(object other):
    """
    Interpret the input as String
    """
    return String(other)


cdef class String(str):
    def to_camel_case(self):
        cdef cpp.String base = cpp.String(<const PyObject *>self)
        cdef BuiltinMethod bm = BuiltinMethod.new_with_baseptr(self, 'to_camel_case', base._native_ptr())

        return String(bm())

    def to_pascal_case(self):
        cdef cpp.String base = cpp.String(<const PyObject *>self)
        cdef BuiltinMethod bm = BuiltinMethod.new_with_baseptr(self, 'to_pascal_case', base._native_ptr())

        return String(bm())

    def to_snake_case(self):
        cdef cpp.String base = cpp.String(<const PyObject *>self)
        cdef BuiltinMethod bm = BuiltinMethod.new_with_baseptr(self, 'to_snake_case', base._native_ptr())

        return String(bm())

    to_upper = str.upper
    to_lower = str.lower

    def is_empty(self):
        return not self

    def path_join(self, filename):
        return String('/'.join((self, filename)))  # FIXME: use os.sep?

    # TODO: All documented methods


# NOTE: By default conversions are to the ordinary Python str

cdef public object string_to_pyobject(const cpp.String &p_string):
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(p_string._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(p_string._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return PyUnicode_FromWideChar(wstr.get_data(), len)


cdef public object typeptr_string_to_pyobject(const void *p_string):
    cdef cpp.String ret = (<cpp.String *>p_string)[0]
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(ret._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(ret._native_ptr(), wstr.ptrw(), len)
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
    cdef bytes tmp

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars(r_ret, wstr)
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        gdextension_interface_string_new_with_utf8_chars(r_ret, cstr)
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        gdextension_interface_string_new_with_utf8_chars(r_ret, cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ String" % p_obj)
        r_ret[0] = cpp.String()


cdef public void typeptr_string_from_pyobject(object p_obj, void *r_ret) noexcept:
    cdef const wchar_t *wstr
    cdef const char *cstr
    cdef bytes tmp

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars(<cpp.String *>r_ret, wstr)
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        gdextension_interface_string_new_with_utf8_chars(<cpp.String *>r_ret, cstr)
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        gdextension_interface_string_new_with_utf8_chars(<cpp.String *>r_ret, cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ String" % p_obj)
        (<cpp.String *>r_ret)[0] = cpp.String()


cdef public void variant_string_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef const wchar_t * wstr
    cdef const char * cstr
    cdef bytes tmp

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        r_ret[0] = cpp.Variant(wstr)
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        r_ret[0] = cpp.Variant(cstr)
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        r_ret[0] = cpp.Variant(cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ String" % p_obj)
        cstr = ""
        r_ret[0] = cpp.Variant(cstr)
