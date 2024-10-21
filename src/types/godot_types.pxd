from gdextension_interface cimport *
from binding cimport *
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stddef cimport wchar_t
cimport godot_cpp as cpp

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy


cdef public object bool_to_pyobject(bint p_bool)
cdef public object variant_bool_to_pyobject(const cpp.Variant &v)
cdef public void bool_from_pyobject(object p_obj, bint *r_ret) noexcept
cdef public void *variant_bool_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public object int_to_pyobject(int64_t p_int)
cdef public object variant_int_to_pyobject(const cpp.Variant &v)
cdef public void int_from_pyobject(object p_obj, int64_t *r_ret) noexcept
cdef public void variant_int_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public object float_to_pyobject(double p_float)
cdef public object variant_float_to_pyobject(const cpp.Variant &v)
cdef public void float_from_pyobject(object p_obj, double *r_ret) noexcept
cdef public void variant_float_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef class String(str): pass
cdef public object string_to_pyobject(const cpp.String &p_string)
cdef public object variant_string_to_pyobject(const cpp.Variant &v)
cdef public void string_from_pyobject(object p_obj, cpp.String *r_ret) noexcept
cdef public void variant_string_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cpdef asvector2(data, dtype=*)
cpdef asvector2i(data, dtype=*)
cdef public object vector2_to_pyobject(cpp.Vector2 &vec)
cdef public object vector2i_to_pyobject(cpp.Vector2i &vec)
cdef public object variant_vector2_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector2i_to_pyobject(const cpp.Variant &v)
cdef public void vector2_from_pyobject(object obj, cpp.Vector2 *r_ret) noexcept
cdef public void vector2i_from_pyobject(object obj, cpp.Vector2i *r_ret) noexcept
cdef public void variant_vector2_from_pyobject(object obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector2i_from_pyobject(object obj, cpp.Variant *r_ret) noexcept

cpdef asrect2(data, dtype=*)
cpdef asrect2i(data, dtype=*)
