from gdextension_interface cimport *
from binding cimport *
from libc.stdint cimport *
from libc.stddef cimport wchar_t
cimport cpython
cimport godot_cpp as cpp

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy


import numpy as np


ctypedef object (*variant_to_pyobject_func_t)(const cpp.Variant &)
ctypedef void (*variant_from_pyobject_func_t)(object, cpp.Variant *) noexcept


ctypedef fused number_t:
    float
    double
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t


ctypedef fused memory_view_t:
    float  [:]
    float  [:, :]
    double [:]
    double [:, :]
    int8_t  [:]
    int8_t  [:, :]
    int16_t [:]
    int16_t [:, :]
    int32_t [:]
    int32_t [:, :]
    int64_t [:]
    int64_t [:, :]
    uint8_t  [:]
    uint8_t  [:, :]
    uint16_t [:]
    uint16_t [:, :]
    uint32_t [:]
    uint32_t [:, :]
    uint64_t [:]
    uint64_t [:, :]


cdef inline bint issubscriptable(object obj):
    return numpy.PyArray_Check(obj) or cpython.PySequence_Check(obj)


cdef inline bint isstring_dtype(object dtype):
    return np.issubdtype(dtype, np.character) or np.issubdtype(dtype, np.dtypes.StringDType) \
           or np.issubdtype(dtype, np.dtypes.BytesDType)


cdef inline void carr_view_from_pyobject(object obj, memory_view_t carr_view, dtype, size_t size,
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

    cdef memory_view_t pyarr_view = arr

    if slice_from != 0 and slice_to != -1:
        carr_view[:] = pyarr_view[slice_from:slice_to]
    if slice_from != 0:
        carr_view[:] = pyarr_view[slice_from:]
    elif slice_to != -1:
        carr_view[:] = pyarr_view[:slice_to]
    else:
        carr_view[:] = pyarr_view

    return


cdef public object variant_nil_to_pyobject(const cpp.Variant &v)
cdef public void variant_nil_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

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

cdef public object vector2_to_pyobject(cpp.Vector2 &vec)
cdef public object vector2i_to_pyobject(cpp.Vector2i &vec)
cdef public object variant_vector2_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector2i_to_pyobject(const cpp.Variant &v)
cdef public void vector2_from_pyobject(object p_obj, cpp.Vector2 *r_ret) noexcept
cdef public void vector2i_from_pyobject(object p_obj, cpp.Vector2i *r_ret) noexcept
cdef public void variant_vector2_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector2i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object rect2_to_pyobject(cpp.Rect2 &rect)
cdef public object rect2i_to_pyobject(cpp.Rect2i &rect)
cdef public object variant_rect2_to_pyobject(const cpp.Variant &v)
cdef public object variant_rect2i_to_pyobject(const cpp.Variant &v)
cdef public void rect2_from_pyobject(object p_obj, cpp.Rect2 *r_ret) noexcept
cdef public void rect2i_from_pyobject(object p_obj, cpp.Rect2i *r_ret) noexcept
cdef public void variant_rect2_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_rect2i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object vector3_to_pyobject(cpp.Vector3 &vec)
cdef public object vector3i_to_pyobject(cpp.Vector3i &vec)
cdef public object variant_vector3_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector3i_to_pyobject(const cpp.Variant &v)
cdef public void vector3_from_pyobject(object p_obj, cpp.Vector3 *r_ret) noexcept
cdef public void vector3i_from_pyobject(object p_obj, cpp.Vector3i *r_ret) noexcept
cdef public void variant_vector3_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector3i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object transform2d_to_pyobject(const cpp.Transform2D &t)
cdef public object variant_transform2d_to_pyobject(const cpp.Variant &v)
cdef public void transform2d_from_pyobject(object p_obj, cpp.Transform2D *r_ret) noexcept
cdef public void variant_transform2d_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object vector4_to_pyobject(cpp.Vector4 &vec)
cdef public object vector4i_to_pyobject(cpp.Vector4i &vec)
cdef public object variant_vector4_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector4i_to_pyobject(const cpp.Variant &v)
cdef public void vector4_from_pyobject(object p_obj, cpp.Vector4 *r_ret) noexcept
cdef public void vector4i_from_pyobject(object p_obj, cpp.Vector4i *r_ret) noexcept
cdef public void variant_vector4_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector4i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object plane_to_pyobject(cpp.Plane &plane)
cdef public object variant_plane_to_pyobject(const cpp.Variant &v)
cdef public void plane_from_pyobject(object p_obj, cpp.Plane *r_ret) noexcept
cdef public void variant_plane_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object quaternion_to_pyobject(cpp.Quaternion &q)
cdef public object variant_quaternion_to_pyobject(const cpp.Variant &v)
cdef public void quaternion_from_pyobject(object p_obj, cpp.Quaternion *r_ret) noexcept
cdef public void variant_quaternion_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object aabb_to_pyobject(cpp._AABB &p_aabb)
cdef public object variant_aabb_to_pyobject(const cpp.Variant &v)
cdef public void aabb_from_pyobject(object p_obj, cpp._AABB *r_ret) noexcept
cdef public void variant_aabb_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept


cdef class StringName(str):
    # TODO: Try to use GDExtension API directly without godot-cpp objects
    cdef cpp.StringName _base
    cdef void *ptr(self)

cdef public object string_name_to_pyobject(const cpp.StringName &p_val)
cdef public object variant_string_name_to_pyobject(const cpp.Variant &v)
cdef public void string_name_from_pyobject(object p_obj, cpp.StringName *r_ret) noexcept
cdef public void variant_string_name_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object node_path_to_pyobject(const cpp.NodePath &p_val)
cdef public object variant_node_path_to_pyobject(const cpp.Variant &v)
cdef public void node_path_from_pyobject(object p_obj, cpp.NodePath *r_ret) noexcept
cdef public void variant_node_path_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef class RID:
    # TODO: Try to use GDExtension API directly without godot-cpp objects
    cdef cpp._RID _base

    @staticmethod
    cdef RID from_cpp_rid(const cpp._RID &p_val)

cdef public object rid_to_pyobject(const cpp._RID &p_val)
cdef public object variant_rid_to_pyobject(const cpp.Variant &v)
cdef public void rid_from_pyobject(object p_obj, cpp._RID *r_ret) noexcept
cdef public void variant_rid_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object dictionary_to_pyobject(const cpp.Dictionary &p_val)
cdef public object variant_dictionary_to_pyobject(const cpp.Variant &v)
cdef public void dictionary_from_pyobject(object p_obj, cpp.Dictionary *r_ret) noexcept
cdef public void variant_dictionary_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object array_to_pyobject(const cpp.Array &p_arr)
cdef public object variant_array_to_pyobject(const cpp.Variant &v)
cdef public void array_from_pyobject(object p_obj, cpp.Array *r_ret) noexcept
cdef public void variant_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object packed_byte_array_to_pyobject(cpp.PackedByteArray &p_arr)
cdef public object packed_int32_array_to_pyobject(cpp.PackedInt32Array &p_arr)
cdef public object packed_int64_array_to_pyobject(cpp.PackedInt64Array &p_arr)
cdef public object packed_float32_array_to_pyobject(cpp.PackedFloat32Array &p_arr)
cdef public object packed_float64_array_to_pyobject(cpp.PackedFloat64Array &p_arr)
cdef public object packed_string_array_to_pyobject(const cpp.PackedStringArray &p_arr)
cdef public object variant_packed_byte_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_int32_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_int64_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_float32_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_float64_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_string_array_to_pyobject(const cpp.Variant &v)
cdef public void packed_byte_array_from_pyobject(object p_obj, cpp.PackedByteArray *r_ret) noexcept
cdef public void packed_int32_array_from_pyobject(object p_obj, cpp.PackedInt32Array *r_ret) noexcept
cdef public void packed_int64_array_from_pyobject(object p_obj, cpp.PackedInt64Array *r_ret) noexcept
cdef public void packed_float32_array_from_pyobject(object p_obj, cpp.PackedFloat32Array *r_ret) noexcept
cdef public void packed_float64_array_from_pyobject(object p_obj, cpp.PackedFloat64Array *r_ret) noexcept
cdef public void packed_string_array_from_pyobject(object p_obj, cpp.PackedStringArray *r_ret) noexcept
cdef public void variant_packed_byte_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_int32_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_int64_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_float32_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_float64_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_string_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept


cdef variant_to_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_to_pyobject_funcs
cdef variant_from_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_from_pyobject_funcs

cdef cpp.VariantType pytype_to_variant_type(type p_type) noexcept
cdef cpp.VariantType pyobject_to_variant_type(object p_obj) noexcept

cdef public object variant_to_pyobject(const cpp.Variant &v)
cdef public void variant_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
