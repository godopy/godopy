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

import numpy as np

cdef bint issubscriptable(object obj) noexcept


ctypedef fused number_t:
    float
    double
    int8_t
    int16_t
    int32_t
    int64_t


cdef inline void carr_view_from_pyobject(object obj, number_t [:] carr_view, dtype, size_t size,
                                         int slice_from=0, int slice_to=-1, bint can_cast=False):
    cdef numpy.ndarray arr

    if not issubscriptable(obj):
        msg = "Cannot convert %r to C++ object, %s objects are unsubscriptable"
        cpp.UtilityFunctions.push_error(msg % (obj, type(obj)))
        return

    elif len(obj) != size:
        msg = "Cannot convert %r to C++ object, expected an object of length %d"
        cpp.UtilityFunctions.push_error(msg % (obj, size))
        return

    if isinstance(obj, numpy.ndarray):
        if obj.dtype == dtype:
            arr = obj
        else:
            if not can_cast:
                msg = "Cast from %r to %r during Godot math type convertion"
                cpp.UtilityFunctions.push_warning(msg % (obj.dtype, dtype))
            arr = obj.astype(dtype)
    else:
        arr = np.array(obj, dtype=dtype)

    cdef number_t [:] pyarr_view = arr

    if slice_from != 0 and slice_to != -1:
        carr_view[:] = pyarr_view[slice_from:slice_to]
    if slice_from != 0:
        carr_view[:] = pyarr_view[slice_from:]
    elif slice_to != -1:
        carr_view[:] = pyarr_view[:slice_to]
    else:
        carr_view[:] = pyarr_view

    return


cdef public object bool_to_pyobject(uint8_t p_bool)
cdef public object variant_bool_to_pyobject(const cpp.Variant &v)
cdef public void bool_from_pyobject(object p_obj, uint8_t *r_ret) noexcept
cdef public void variant_bool_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
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
cdef public object rect2_to_pyobject(cpp.Rect2 &rect)
cdef public object rect2i_to_pyobject(cpp.Rect2i &rect)
cdef public object variant_rect2_to_pyobject(const cpp.Variant &v)
cdef public object variant_rect2i_to_pyobject(const cpp.Variant &v)
cdef public void rect2_from_pyobject(object obj, cpp.Rect2 *r_ret) noexcept
cdef public void rect2i_from_pyobject(object obj, cpp.Rect2i *r_ret) noexcept
cdef public void variant_rect2_from_pyobject(object obj, cpp.Variant *r_ret) noexcept
cdef public void variant_rect2i_from_pyobject(object obj, cpp.Variant *r_ret) noexcept
