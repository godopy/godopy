"""
Python versions of Godot Variant types.

Default types:
    Nil -> None
    bool -> bool
    int -> int
    float -> float
    String -> str
    StringName -> godot_types.StringName, subclass of str, wraps C++ StringName
    NodePath -> godot_types.NodePath, subclass of pathlib.PurePosixPath
    RID -> godot_types.RID, custom class, wraps C++ RID
    Callable -> godot_types.Callable, custom class, wraps C++ Callable
    Signal -> godot_types.Signal, custom class, wraps C++ Signal
    Dictionary -> dict
    Array -> list
    TypedArray -> godot_types.Array, subclass of numpy.ndarray
    PackedStringArray -> list
    <AnyOtherType> -> godot_types.<AnyOtherType>, subclass of numpy.ndarray

Extended types:
    String -> godot_types.String, subclass of str
    Array -> godot_types.Array
    PackedStringArray -> godot_types.PackedStringArray, subclass of numpy.dnarray
"""
from cpython cimport (
    PyObject, ref,
    PyUnicode_AsWideCharString, PyUnicode_FromWideChar,
    PyBytes_AsString,
    PyBool_Check, PyLong_Check, PyFloat_Check, PyUnicode_Check, PyBytes_Check,
    PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release,
    PySequence_Check, PySequence_Size, PySequence_GetItem,
    PyMapping_Check, PyMapping_Keys, PyObject_GetItem,
    PyIndex_Check, PyNumber_Check, PyComplex_Check,
    PyObject_IsTrue,
    PyList_New, PyList_SET_ITEM, PyList_GET_ITEM
)
from libcpp.cast cimport *
from cpython.bytearray cimport PyByteArray_Check
from cython.view cimport array as cvarray
from gdextension cimport (
    BuiltinMethod, Object,
    object_to_pyobject, variant_object_to_pyobject,
    object_from_pyobject, variant_object_from_pyobject,
    variant_type_to_str, str_to_variant_type
)
from numpy cimport (
    PyArray_New, PyArray_Check, PyArray_TYPE,
    npy_intp,
    NPY_BYTE, NPY_INT16, NPY_INT32, NPY_INT64, NPY_FLOAT32, NPY_FLOAT64,
    NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_WRITEABLE,
)

import sys
import pathlib

import numpy as np


__all__ = [
    'Nil',

    'bool',
    'int',
    'float',

    "as_string",
    'String',

    'as_vector2',
    'as_vector2i',

    'Vector2',
    'Vector2i',
    'Size2',
    'Size2i',

    'as_rect2',
    'as_rect2i',

    'Rect2',
    'Rect2i',

    'as_vector3',
    'as_vector3i',
    'Vector3',
    'Vector3i',

    'as_transform2d',
    'Transform2D',

    'as_vector4',
    'as_vector4i',
    'Vector4',
    'Vector4i',

    'as_plane',
    'Plane',

    'as_quaternion',
    'Quaternion',

    'as_aabb',
    'AABB',

    'as_basis',
    'Basis',

    'as_transform2d',
    'Transform3D',

    'as_projection',
    'Projection',

    'as_color',
    'Color',

    'as_string_name',
    'StringName',

    'as_node_path',
    'NodePath',

    'RID',

    # Object is in gdextension module

    'Callable',
    'Signal',

    'Dictionary',

    'as_array',
    'Array',

    'as_packed_byte_array',
    'as_packed_int32_array',
    'as_packed_int64_array',
    'as_packed_float32_array',
    'as_packed_float64_array',
    'as_packed_string_array',
    'as_packed_vector2_array',
    'as_packed_vector3_array',
    'as_packed_color_array',
    'as_packed_vector4_array',
    'PackedByteArray',
    'PackedInt32Array',
    'PackedInt64Array',
    'PackedFloat32Array',
    'PackedFloat64Array',
    'PackedStringArray',
    'PackedVector2Array',
    'PackedVector3Array',
    'PackedColorArray',
    'PackedVector4Array'
]


cdef object PyArraySubType_NewFromBase(type subtype, numpy.ndarray base):
    cdef numpy.ndarray arr = PyArray_New(subtype, base.ndim, base.shape, PyArray_TYPE(base), NULL, base.data, 0,
                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, base)
    ref.Py_INCREF(base)
    numpy.PyArray_SetBaseObject(arr, base)

    return arr


cdef class _PackedArrayWrapper:
    cdef cvarray arr
    cdef npy_intp size
    cdef npy_intp[2] shape
    cdef bint ro
    cdef npy_intp ndim
    cdef int type_num

    def __cinit__(self):
        self.ro = False

    def __init__(self):
        raise TypeError("Cannot init %r" % self.__class__)


cdef object PyArraySubType_NewFromWrapperBase(type subtype, _PackedArrayWrapper base):
    cdef numpy.ndarray arr = PyArray_New(subtype, base.ndim, base.shape, base.type_num, NULL, base.arr.data, 0,
                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, base)

    ref.Py_INCREF(base)
    numpy.PyArray_SetBaseObject(arr, base)

    return arr


def _check_numeric_scalar(scalar, arg_name=None):
    if not isinstance(scalar, (np.number, int, float, np.bool_)):
        argument = 'argument %r' % arg_name if arg_name else 'scalar'
        raise ValueError("Invalid %s %r in data" % (arg_name, scalar))


include "includes/atomic.pxi"
include "includes/vector2.pxi"
include "includes/rect2.pxi"
include "includes/vector3.pxi"
include "includes/transform2d.pxi"
include "includes/vector4.pxi"
include "includes/plane.pxi"
include "includes/quaternion.pxi"
include "includes/aabb.pxi"
include "includes/basis.pxi"
include "includes/transform3d.pxi"
include "includes/projection.pxi"
include "includes/color.pxi"
include "includes/misc.pxi"
include "includes/packed_arrays.pxi"


cdef type NoneType = type(None)

cdef dict _pytype_to_vartype = {
    NoneType: cpp.NIL,
    bool: cpp.BOOL,
    np.bool_: cpp.BOOL,
    int: cpp.INT,
    np.int8: cpp.INT,
    np.int16: cpp.INT,
    np.int32: cpp.INT,
    np.int64: cpp.INT,
    np.uint8: cpp.INT,
    np.uint16: cpp.INT,
    np.uint32: cpp.INT,
    np.uint64: cpp.INT,
    float: cpp.FLOAT,
    np.float16: cpp.FLOAT,
    np.float32: cpp.FLOAT,
    np.float64: cpp.FLOAT,
    str: cpp.STRING,
    bytes: cpp.STRING,
    String: cpp.STRING,
    np.str_: cpp.STRING,
    np.bytes_: cpp.STRING,
    np.void: cpp.PACKED_BYTE_ARRAY,
    complex: cpp.VECTOR2,
    Vector2: cpp.VECTOR2,
    Vector2i: cpp.VECTOR2I,
    Rect2: cpp.RECT2,
    Rect2i: cpp.RECT2I,
    Vector3: cpp.VECTOR3,
    Vector3i: cpp.VECTOR3I,
    Transform2D: cpp.TRANSFORM2D,
    Vector4: cpp.VECTOR4,
    Vector4i: cpp.VECTOR4I,
    Plane: cpp.PLANE,
    Quaternion: cpp.QUATERNION,
    AABB: cpp.AABB,
    Basis: cpp.BASIS,
    Transform3D: cpp.TRANSFORM3D,
    Projection: cpp.PROJECTION,
    Color: cpp.COLOR,
    StringName: cpp.STRING_NAME,
    NodePath: cpp.NODE_PATH,
    pathlib.PurePosixPath: cpp.NODE_PATH,
    RID: cpp.RID,
    Object: cpp.OBJECT,
    Callable: cpp.CALLABLE,
    Signal: cpp.SIGNAL,
    dict: cpp.DICTIONARY,
    list: cpp.ARRAY,
    tuple: cpp.ARRAY,
    bytearray: cpp.PACKED_BYTE_ARRAY,
    PackedByteArray: cpp.PACKED_BYTE_ARRAY,
    PackedInt32Array: cpp.PACKED_INT32_ARRAY,
    PackedInt64Array: cpp.PACKED_INT64_ARRAY,
    PackedFloat32Array: cpp.PACKED_FLOAT32_ARRAY,
    PackedFloat64Array: cpp.PACKED_FLOAT64_ARRAY,
    PackedStringArray: cpp.PACKED_STRING_ARRAY,
    PackedVector2Array: cpp.PACKED_VECTOR2_ARRAY,
    PackedVector3Array: cpp.PACKED_VECTOR3_ARRAY,
    PackedColorArray: cpp.PACKED_COLOR_ARRAY,
    PackedVector4Array: cpp.PACKED_VECTOR4_ARRAY
}


cdef variant_to_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_to_pyobject_funcs = [
    variant_nil_to_pyobject,
    variant_bool_to_pyobject,
    variant_int_to_pyobject,
    variant_float_to_pyobject,
    variant_string_to_pyobject,
    variant_vector2_to_pyobject,
    variant_vector2i_to_pyobject,
    variant_rect2_to_pyobject,
    variant_rect2i_to_pyobject,
    variant_vector3_to_pyobject,
    variant_vector3i_to_pyobject,
    variant_transform2d_to_pyobject,
    variant_vector4_to_pyobject,
    variant_vector4i_to_pyobject,
    variant_plane_to_pyobject,
    variant_quaternion_to_pyobject,
    variant_aabb_to_pyobject,
    variant_basis_to_pyobject,
    variant_transform3d_to_pyobject,
    variant_projection_to_pyobject,
    variant_color_to_pyobject,
    variant_string_name_to_pyobject,
    variant_node_path_to_pyobject,
    variant_rid_to_pyobject,
    variant_object_to_pyobject,
    variant_callable_to_pyobject,
    variant_signal_to_pyobject,
    variant_dictionary_to_pyobject,
    variant_array_to_pyobject,
    variant_packed_byte_array_to_pyobject,
    variant_packed_int32_array_to_pyobject,
    variant_packed_int64_array_to_pyobject,
    variant_packed_float32_array_to_pyobject,
    variant_packed_float64_array_to_pyobject,
    variant_packed_string_array_to_pyobject,
    variant_packed_vector2_array_to_pyobject,
    variant_packed_vector3_array_to_pyobject,
    variant_packed_color_array_to_pyobject,
    variant_packed_vector4_array_to_pyobject,
]


cdef typeptr_to_pyobject_func_t[<int>cpp.VARIANT_MAX] typeptr_to_pyobject_funcs = [
    typeptr_nil_to_pyobject,
    typeptr_bool_to_pyobject,
    typeptr_int_to_pyobject,
    typeptr_float_to_pyobject,
    typeptr_string_to_pyobject,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
]



cdef variant_from_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_from_pyobject_funcs = [
    variant_nil_from_pyobject,
    variant_bool_from_pyobject,
    variant_int_from_pyobject,
    variant_float_from_pyobject,
    variant_string_from_pyobject,
    variant_vector2_from_pyobject,
    variant_vector2i_from_pyobject,
    variant_rect2_from_pyobject,
    variant_rect2i_from_pyobject,
    variant_vector3_from_pyobject,
    variant_vector3i_from_pyobject,
    variant_transform2d_from_pyobject,
    variant_vector4_from_pyobject,
    variant_vector4i_from_pyobject,
    variant_plane_from_pyobject,
    variant_quaternion_from_pyobject,
    variant_aabb_from_pyobject,
    variant_basis_from_pyobject,
    variant_transform3d_from_pyobject,
    variant_projection_from_pyobject,
    variant_color_from_pyobject,
    variant_string_name_from_pyobject,
    variant_node_path_from_pyobject,
    variant_rid_from_pyobject,
    variant_object_from_pyobject,
    variant_callable_from_pyobject,
    variant_signal_from_pyobject,
    variant_dictionary_from_pyobject,
    variant_array_from_pyobject,
    variant_packed_byte_array_from_pyobject,
    variant_packed_int32_array_from_pyobject,
    variant_packed_int64_array_from_pyobject,
    variant_packed_float32_array_from_pyobject,
    variant_packed_float64_array_from_pyobject,
    variant_packed_string_array_from_pyobject,
    variant_packed_vector2_array_from_pyobject,
    variant_packed_vector3_array_from_pyobject,
    variant_packed_color_array_from_pyobject,
    variant_packed_vector4_array_from_pyobject,
]


cdef typeptr_from_pyobject_func_t[<int>cpp.VARIANT_MAX] typeptr_from_pyobject_funcs = [
    typeptr_nil_from_pyobject,
    typeptr_bool_from_pyobject,
    typeptr_int_from_pyobject,
    typeptr_float_from_pyobject,
    typeptr_string_from_pyobject,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
]


cdef int array_to_vartype(object arr) except -2:
    cdef int vartype = -1

    cdef size_t itemsize = arr.itemsize
    cdef tuple shape = arr.shape
    cdef object dtype = arr.dtype
    cdef size_t ndim = arr.ndim

    if vartype < 0:
        if shape == (2,):
            if np.issubdtype(dtype, np.integer):
                vartype = <int>cpp.VECTOR2I
            else:
                vartype = <int>cpp.VECTOR2
        elif shape == (3,):
            if np.issubdtype(dtype, np.integer):
                vartype = <int>cpp.VECTOR3I
            else:
                vartype = <int>cpp.VECTOR3
        elif shape == (4,):
            if np.issubdtype(dtype, np.integer):
                vartype = <int>cpp.VECTOR4I
            else:
                vartype = <int>cpp.VECTOR4

    if vartype < 0:
        if ndim == 1:
            if vartype < 0:
                if np.issubdtype(dtype, np.integer):
                    if itemsize == 1:
                        vartype = <int>cpp.PACKED_BYTE_ARRAY
                    elif itemsize <= 4:
                        vartype = <int>cpp.PACKED_INT32_ARRAY
                    else:
                        vartype = <int>cpp.PACKED_INT64_ARRAY
                elif np.issubdtype(dtype, np.floating):
                    if itemsize <= 4:
                        vartype = <int>cpp.PACKED_FLOAT32_ARRAY
                    else:
                        vartype = <int>cpp.PACKED_FLOAT64_ARRAY
                elif isstring_dtype(dtype):
                    vartype = <int>cpp.PACKED_STRING_ARRAY
        elif ndim == 2:
            ndim2_size = shape[2]
            if ndim2_size == 2:
                vartype = <int>cpp.PACKED_VECTOR2_ARRAY
            elif ndim2_size == 3:
                vartype = <int>cpp.PACKED_VECTOR3_ARRAY
            elif ndim2_size == 4:
                vartype = <int>cpp.PACKED_VECTOR4_ARRAY
    
    if vartype < 0:
        cpp.UtilityFunctions.push_warning("Unknown array %r" % arr)

    return vartype


cdef cpp.VariantType pytype_to_variant_type(type p_type) noexcept:
    return _pytype_to_vartype.get(p_type, <int>cpp.OBJECT )


cdef cpp.VariantType pyobject_to_variant_type(object p_obj) noexcept:
    cdef int vartype = _pytype_to_vartype.get(type(p_obj), -1)
    cdef numpy.ndarray arr

    if vartype < 0:
        if PyBool_Check(p_obj) or isinstance(p_obj, np.bool_):
            vartype = <int>cpp.BOOL
        elif PyLong_Check(p_obj) or PyIndex_Check(p_obj) or isinstance(p_obj, np.integer):
            vartype = <int>cpp.INT
        elif PyFloat_Check(p_obj) or PyNumber_Check(p_obj) or isinstance(p_obj, np.floating):
            vartype = <int>cpp.FLOAT
        elif PyUnicode_Check(p_obj) or PyBytes_Check(p_obj) or isinstance(p_obj, np.character):
            vartype = <int>cpp.STRING
        elif PyComplex_Check(p_obj):
            vartype = <int>cpp.VECTOR2
        elif PyByteArray_Check(p_obj):
            vartype = <int>cpp.PACKED_BYTE_ARRAY
        elif PyArray_Check(p_obj):
            arr = p_obj
            vartype = array_to_vartype(arr)
        elif PyObject_CheckBuffer(p_obj):
            arr = np.array(p_obj)
            vartype = array_to_vartype(arr)
        elif PySequence_Check(p_obj):
            vartype = <int>cpp.ARRAY
        elif PyMapping_Check(p_obj):
            vartype = <int>cpp.DICTIONARY
        else:
            vartype = <int>cpp.OBJECT        

    return <cpp.VariantType>vartype


cdef public object variant_to_pyobject(const cpp.Variant &v):
    cdef int vartype = <int>v.get_type()
    cdef variant_to_pyobject_func_t func = variant_to_pyobject_funcs[vartype]

    return func(v)


cdef public void variant_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int vartype = <int>pyobject_to_variant_type(p_obj)
    cdef variant_from_pyobject_func_t func
    cdef str msg

    assert vartype >= 0 and vartype < <int>cpp.VARIANT_MAX, "incorrect vartype %d" % vartype

    func = variant_from_pyobject_funcs[vartype]
    func(p_obj, r_ret)
