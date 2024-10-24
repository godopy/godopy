"""
Python versions of Godot Variant types
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
from cpython.bytearray cimport PyByteArray_Check
from cython.view cimport array as cvarray
from gdextension cimport (
    BuiltinMethod,
    Object, Callable, Signal,
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
    'Nil',  # None

    'bool',  # bool or np.bool or ndarray as array(x, dtype=np.bool), shape = ()
    'int',  # int or np.int64 or np.int32 or np.int8 or ndarray as array(x, dtype=intN), shape = ()
    'float',  # float or np.float64 or np.float32 or ndarray as array(x, dtype=floatN), shape = ()

    "as_string",
    'String',  # subclass of str or str or bytes

    'as_vector2',
    'as_vector2i',

    'Vector2',  # subtype of ndarray as array([x, y], dtype=float32), shape = (2,)
    'Vector2i',  # subtype of ndarray as array([x, y], dtype=int32), shape = (2,)
    'Size2',  # same as Vector2, but has width and height attributes
    'Size2i',  # same as Vector2i, but has width and height attributes

    'as_rect2',
    'as_rect2i',

    'Rect2',  # subtype of ndarray as array([x, y, width, height], dtype=float32), shape = (4,), slices: Vector2, Size2
    'Rect2i',  # subtype of ndarray as array([x, y, width, height], dtype=int32), shape = (4,), slices: Vector2i, Size2i

    'as_vector3',
    'as_vector3i',
    'Vector3',  # TODO subtype of ndarray as array([x, y, z], dtype=float32), shape = (3,)
    'Vector3i',  # TODO subtype of ndarray as array([x, y, z], dtype=int32), shape = (3,)

    'as_transform2d',
    'Transform2D',  # TODO subtype of ndarray as array([[xx, xy],
                    #                                   [yx, yy],
                    #                                   [zx, zy]], dtype=float32), shape = (3, 2)

    'as_vector4',
    'as_vector4i',
    'Vector4',  # TODO subtype of ndarray as array([x, y, z, w], dtype=float32), shape = (4,)
    'Vector4i',  # TODO subtype of ndarray as array([x, y, z, w], dtype=int32), shape = (4,)

    'as_plane',
    'Plane',  # TODO: subtype of ndarray as array([x, y, z, d], dtype=float32), shape = (4,)
              # slices: Vector3, float
    'Quaternion',  # TODO subtype of ndarray([x, y, z, w], dtype=float32), shape = (4,)
                   # or _godot_type_tuples.Quaternion()
    'AABB',  # TODO: subtype of ndarray as array([[x, y, z],
             #                                    [x, y, z]], dtype=float32), shape = (2, 3)
    'Basis',  # TODO subtype of ndarray as array([[xx, xy, xz],
              #                                   [yx, yy, yz],
              #                                   [zx, zy, zz]], dtype=float32), shape = (3, 3)
    'Transform3D',  # TODO subtype of ndarray as array([[xx, xy, xz],
                    #                                   [yx, yy, yz],
                    #                                   [zx, zy, zz]
                    #                                   [ox, oy, oz]], dtype=float32), shape = (4, 3)
                    # slices: Basis, Vector3
    'Projection',  # TODO subtype of ndarray as array([[xx, xy, xz, xw],
                   #                                   [yx, yy, yz, yw],
                   #                                   [zx, zy, zz, zw],
                   #                                   [wx, wy, wx, ww]], dtype=float32), shape = (4, 4)
    'Color',    # TODO subtype of ndarray as array([x, y, z, w], dtype=float32), shape = (4,)
    'StringName',  # sublcass of str, cpp.StringName wrapper
    'NodePath',  # sublcass of pathlib.PurePosixPath
    'RID',  # cpp._RID wrapper

    # Object, Callable, Signal are in gdextension module

    'Dictionary',  # dict

    'as_array',
    'Array',  # list or subtype of ndarray as array([...], dtype=np.object_), shape = (N,)

    'as_packed_byte_array',
    'as_packed_int32_array',
    'as_packed_int64_array',
    'as_packed_float32_array',
    'as_packed_float64_array',
    'as_packed_string_array',
    'PackedByteArray',  # is bytearray or TODO subtype of ndarray as array([...], dtype=np.int8)
    'PackedInt32Array',  # TODO subtype of ndarray as array([...], dtype=np.int32), shape = (N,)
    'PackedInt64Array',  # TODO subtype of ndarray as array([...], dtype=np.int64), shape = (N,)
    'PackedFloat32Array',  # TODO subtype of ndarray as array([...], dtype=np.float32), shape = (N,)
    'PackedFloat64Array',  # TODO subtype of ndarray as array([...], dtype=np.float64), shape = (N,)
    'PackedStringArray',  # TODO subtype of ndarray as array([...], dtype=StringDType())

    'PackedVector2Array',  # TODO subtype of ndarray as array([[x, y], ...], dtype=np.float32), shape = (N, 2)
    'PackedVector3Array',  # TODO subtype of ndarray as array([[x, y, z], ...], dtype=np.float32), shape = (N, 3)
    'PackedColorArray',  # TODO subtype of ndarray as array([[r, g, b, a], ...], dtype=np.float32), shape = (N, 4)
    'PackedVector4Array'  # TODO subtype of ndarray as array([[x, y, z, w], ...], dtype=np.float32), shape = (N, 4)
]


cdef object PyArraySubType_NewFromBase(type subtype, numpy.ndarray base):
    cdef numpy.ndarray arr = PyArray_New(subtype, base.ndim, base.shape, PyArray_TYPE(base), NULL, base.data, 0,
                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, base)
    ref.Py_INCREF(base)
    numpy.PyArray_SetBaseObject(arr, base)

    return arr

cdef object PyArraySubType_NewFromDataAndBase(type subtype, cvarray cvarr, int nd, const npy_intp *dims,
                                              int type_num, object base):
    cdef int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE
    cdef numpy.ndarray arr = PyArray_New(subtype, nd, dims, type_num, NULL, cvarr.data, 0, flags, base)

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
include "includes/packed_array1d.pxi"
include "includes/packed_array2d.pxi"


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
    NULL,  # vector3
    NULL,  # vector3i
    NULL,  # transform2d
    NULL,  # vector4
    NULL,  # vector4i
    NULL,  # plane
    NULL,  # quaternion
    NULL,  # aabb
    NULL,  # basis
    NULL,  # transform3d
    NULL,  # projection
    NULL,  # color
    variant_string_name_to_pyobject,
    variant_node_path_to_pyobject,
    variant_rid_to_pyobject,
    variant_object_to_pyobject,
    NULL,  # callable
    NULL,  # signal
    variant_dictionary_to_pyobject,
    variant_array_to_pyobject,
    NULL,  # packed_byte_array
    NULL,  # packed_int32_array
    NULL,  # int64
    NULL,  # float32
    NULL,  # float64
    variant_packed_string_array_to_pyobject,
    NULL,  # vector2
    NULL,  # vector3
    NULL,  # color
    NULL  # vector4
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
    NULL,  # vector3
    NULL,  # vector3i
    NULL,  # transform2d
    NULL,  # vector4
    NULL,  # vector4i
    NULL,  # plane
    NULL,  # quaternion
    NULL,  # aabb
    NULL,  # basis
    NULL,  # transform3d
    NULL,  # projection
    NULL,  # color
    variant_string_name_from_pyobject,
    variant_node_path_from_pyobject,
    variant_rid_from_pyobject,
    variant_object_from_pyobject,
    NULL,  # callable
    NULL,  # signal
    variant_dictionary_from_pyobject,
    variant_array_from_pyobject,
    NULL,  # packed_byte_array
    NULL,  # packed_int32_array
    NULL,  # int64
    NULL,  # float32
    NULL,  # float64
    variant_packed_string_array_from_pyobject,
    NULL,  # vector2
    NULL,  # vector3
    NULL,  # color
    NULL  # vector4
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

    if func != NULL:
        return func(v)
    else:
        msg = "NOT IMPLEMENTED: convertion of %r types to Python objects, interpret as Nil/None"
        cpp.UtilityFunctions.push_error(msg % variant_type_to_str(<cpp.VariantType>vartype))


cdef public void variant_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int vartype = <int>pyobject_to_variant_type(p_obj)
    cdef variant_from_pyobject_func_t func
    cdef str msg

    assert vartype >= 0 and vartype < <int>cpp.VARIANT_MAX

    func = variant_from_pyobject_funcs[vartype]
    if func != NULL:
        func(p_obj, r_ret)
    else:
        msg = "NOT IMPLEMENTED: convertion of %r types from Python objects, interpret as None/Nil"
        cpp.UtilityFunctions.push_error(msg % variant_type_to_str(<cpp.VariantType>vartype))
        variant_nil_from_pyobject(None, r_ret)
