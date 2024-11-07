Nil = None


cdef public object variant_nil_to_pyobject(const cpp.Variant &v):
    return None


cdef public void variant_nil_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    # if p_obj is not None:
    #     cpp.UtilityFunctions.push_error("'None' is required, got %r" % type(p_obj))
    gdextension_interface_variant_new_nil(r_ret)


bool = bool


cdef public object bool_to_pyobject(const uint8_t p_bool):
    return p_bool != 0


cdef public object variant_bool_to_pyobject(const cpp.Variant &v):
    return gdextension_interface_variant_booleanize(v._native_ptr()) != 0


cdef public void bool_from_pyobject(object p_obj, uint8_t *r_ret) noexcept:
    r_ret[0] = <uint8_t>PyObject_IsTrue(p_obj)


cdef public void variant_bool_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef bint ret = PyObject_IsTrue(p_obj)

    r_ret[0] = cpp.Variant(ret, True)


int = int


cdef public object int_to_pyobject(const int64_t p_int):
    return p_int


cdef public object variant_int_to_pyobject(const cpp.Variant &v):
    cdef int64_t ret = v.to_type[int64_t]()

    return ret


cdef public void int_from_pyobject(object p_obj, int64_t *r_ret) noexcept:
    r_ret[0] = <int64_t>p_obj



cdef public void variant_int_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int64_t ret = <int64_t>p_obj

    cdef cpp.Variant v = cpp.Variant(ret)
    gdextension_interface_variant_new_copy(r_ret, &v)


float = float


cdef public object float_to_pyobject(const double p_float):
    return p_float


cdef public object variant_float_to_pyobject(const cpp.Variant &v):
    cdef double ret = v.to_type[double]()

    return ret


cdef public void float_from_pyobject(object p_obj, double *r_ret) noexcept:
    r_ret[0] = <double>p_obj


cdef public void variant_float_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef double ret = 0.0

    ret = <double>p_obj

    cdef cpp.Variant v = cpp.Variant(ret)
    gdextension_interface_variant_new_copy(r_ret, &v)


def as_string(object other):
    """
    Interpret the input as String
    """
    return String(other)


cdef class String(str):
    def to_camel_case(self):
        cdef cpp.String base = cpp.String(<const PyObject *>self)
        cdef BuiltinMethod bm = BuiltinMethod.new_with_selfptr(self, 'to_camel_case', base._native_ptr())

        return String(bm())

    def to_pascal_case(self):
        cdef cpp.String base = cpp.String(<const PyObject *>self)
        cdef BuiltinMethod bm = BuiltinMethod.new_with_selfptr(self, 'to_pascal_case', base._native_ptr())

        return String(bm())

    def to_snake_case(self):
        cdef cpp.String base = cpp.String(<const PyObject *>self)
        string_from_pyobject(self, &base)
        cdef BuiltinMethod bm = BuiltinMethod.new_with_selfptr(self, 'to_snake_case', base._native_ptr())

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


cdef public object variant_string_to_pyobject(const cpp.Variant &v):
    cdef cpp.String ret
    gdextension_interface_variant_stringify(v._native_ptr(), ret._native_ptr())

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
        gdextension_interface_string_new_with_wide_chars_and_len(r_ret, wstr, len(p_obj))
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        gdextension_interface_string_new_with_utf8_chars_and_len2(r_ret, cstr, len(p_obj))
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        gdextension_interface_string_new_with_utf8_chars_and_len2(r_ret, cstr, len(tmp))
    else:
        p_obj = str(p_obj)
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars_and_len(r_ret, wstr, len(p_obj))


cdef public void variant_string_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef const wchar_t * wstr
    cdef const char * cstr
    cdef bytes tmp
    cdef cpp.String s

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars_and_len(&s, wstr, len(p_obj))
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        gdextension_interface_string_new_with_utf8_chars_and_len2(&s, cstr, len(p_obj))
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        gdextension_interface_string_new_with_utf8_chars_and_len2(&s, cstr, len(tmp))
    else:
        p_obj = str(p_obj)
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars_and_len(&s, wstr, len(p_obj))

    cdef cpp.Variant v = cpp.Variant(s)
    gdextension_interface_variant_new_copy(r_ret, &v)
