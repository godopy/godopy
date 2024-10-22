"""
Python versions of Godot Variant types
"""
from cpython cimport (
    PyObject, ref,
    PyUnicode_AsWideCharString, PyUnicode_FromWideChar,
    PyBytes_AsString,
    PyBool_Check, PyLong_Check, PyFloat_Check, PyUnicode_Check, PyBytes_Check,
    # PyByteArray_Check,
    PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release,
    PySequence_Check, PySequence_Size, PySequence_GetItem,
    PyMapping_Check, PyMapping_Keys, PyObject_GetItem,
    PyIndex_Check, PyNumber_Check,
    PyObject_IsTrue
)
from gdextension cimport (
    BuiltinMethod,
    # Object,
    object_from_pyobject, variant_object_from_pyobject,
    variant_type_to_str, str_to_variant_type
)
from numpy cimport PyArray_New, PyArray_Check, PyArray_TYPE, NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_WRITEABLE

import numpy as np
import _godot_type_tuples as tt

# from gdextension import Callable, Signal

__all__ = [
    'Nil',  # None

    'bool',  # bool or np.bool or ndarray as array(x, dtype=np.bool), shape = ()
    'int',  # int or np.int64 or np.int32 or np.int8 or ndarray as array(x, dtype=intN), shape = ()
    'float',  # float or np.float64 or np.float32 or ndarray as array(x, dtype=floatN), shape = ()
    'String',  # str or bytes

    'asvector2',
    'asvector2i',

    'Vector2',  # subtype of ndarray as array([x, y], dtype=float32), shape = (2,)
    'Vector2i',  # subtype of ndarray as array([x, y], dtype=int32), shape = (2,)
    'Size2',  # same as Vector2, but has width and height attributes
    'Size2i',  # same as Vector2i, but has width and height attributes

    'asrect2',
    'asrect2i',

    'Rect2',  # subtype of ndarray as array([x, y, width, height], dtype=float32), shape = (4,), slices: Vector2, Size2
    'Rect2i',  # subtype of ndarray as array([x, y, width, height], dtype=int32), shape = (4,), slices: Vector2i, Size2i
    'Vector3',  # TODO subtype of ndarray as array([x, y, z], dtype=float32), shape = (3,)
    'Vector3i',  # TODO subtype of ndarray as array([x, y, z], dtype=int32), shape = (3,)
    'Transform2D',  # TODO subtype of ndarray as array([[xx, xy],
                    #                                   [yx, yy],
                    #                                   [zx, zy]], dtype=float32), shape = (3, 2)
    'Vector4',  # TODO subtype of ndarray as array([x, y, z, w], dtype=float32), shape = (4,)
    'Vector4i',  # TODO subtype of ndarray as array([x, y, z, w], dtype=int32), shape = (4,)
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
    'StringName',  # sublcass of str
    'NodePath',  # sublcass of str
    'RID',  # sublcass of int

    # Object, Callable, Signal are in gdextension module

    'Dictionary',  # dict
    'Array',  # list

    'PackedByteArray',  # is bytearray or TODO subtype of ndarray as array([...], dtype=np.int8)
    'PackedInt32Array',  # TODO subtype of ndarray as array([...], dtype=np.int32), shape = (N,)
    'PackedInt64Array',  # TODO subtype of ndarray as array([...], dtype=np.int64), shape = (N,)
    'PackedFloat32Array',  # TODO subtype of ndarray as array([...], dtype=np.float32), shape = (N,)
    'PackedFloat64Array',  # TODO subtype of ndarray as array([...], dtype=np.float64), shape = (N,)
    'PackedStringArray',  # subclass of tuple (itemtype = str)
    'PackedVector2Array',  # TODO subtype of ndarray as array([[x, y], ...], dtype=np.float32), shape = (N, 2)
    'PackedVector3Array',  # TODO subtype of ndarray as array([[x, y, z], ...], dtype=np.float32), shape = (N, 3)
    'PackedColorArray',  # TODO subtype of ndarray as array([[r, g, b, a], ...], dtype=np.float32), shape = (N, 4)
    'PackedVector4Array'  # TODO subtype of ndarray as array([[x, y, z, w], ...], dtype=np.float32), shape = (N, 4)
]


cdef bint issubscriptable(object obj) noexcept:
    return isinstance(obj, (np.ndarray, tuple, list)) or \
           (hasattr(obj, '__len__') and hasattr(obj, '__getitem__'))


cdef object PyArraySubType_NewFromBase(type subtype, numpy.ndarray base):
    cdef numpy.ndarray arr = PyArray_New(subtype, base.ndim, base.shape, PyArray_TYPE(base), NULL,
                                         base.data, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)
    ref.Py_INCREF(base)
    numpy.PyArray_SetBaseObject(arr, base)

    return arr


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
include "includes/packed_1dim_array.pxi"
include "includes/packed_2dim_array.pxi"


cdef type NoneType = type(None)

cdef dict _pytype_to_vartype = {
    NoneType: cpp.NIL,
    bool: cpp.BOOL,
    int: cpp.INT,
    float: cpp.FLOAT,
    str: cpp.STRING,
    bytes: cpp.STRING,
    String: cpp.STRING,
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
    RID: cpp.RID,
    # Object: cpp.OBJECT,
    # Callable: cpp.CALLABLE,
    # Signal: cpp.SIGNAL,
    dict: cpp.DICTIONARY,
    list: cpp.ARRAY,
    tuple: cpp.ARRAY,
    bytearray: cpp.PACKED_BYTE_ARRAY,
    # PackedByteArray: cpp.PACKED_BYTE_ARRAY,
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

ctypedef void (*variant_type_from_pyobject_func_t)(object, cpp.Variant *) noexcept

cdef variant_type_from_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_from_pyobject_funcs = [
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
    NULL,  # string_name
    NULL,  # node_path
    NULL,  # rid
    variant_object_from_pyobject,
    NULL,  # callable
    NULL,  # signal
    NULL,  # dictionary
    NULL,  # array
    NULL,  # packed_byte_array
    NULL,  # packed_int32_array
    NULL,  # int64
    NULL,  # float32
    NULL,  # float64
    NULL,  # string
    NULL,  # vector2
    NULL,  # vector3
    NULL,  # color
    NULL  # vector4
]


cdef int array_to_vartype(object arr) except -2:
    cdef int vartype = -1

    shape = arr.shape
    dtype = arr.dtype
    ndim = arr.ndim
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
            if dtype == np.int8:
                vartype = <int>cpp.PACKED_BYTE_ARRAY
            elif dtype == np.bool:
                vartype = <int>cpp.PACKED_BYTE_ARRAY
            elif dtype == np.int16:
                vartype = <int>cpp.PACKED_INT32_ARRAY
            elif dtype == np.int32:
                vartype = <int>cpp.PACKED_INT32_ARRAY
            elif dtype == np.int64:
                vartype = <int>cpp.PACKED_INT64_ARRAY
            elif dtype == np.float16:
                vartype = <int>cpp.PACKED_FLOAT32_ARRAY
            elif dtype == np.float32:
                vartype = <int>cpp.PACKED_FLOAT32_ARRAY
            elif dtype == np.float64:
                vartype = <int>cpp.PACKED_FLOAT64_ARRAY

            if vartype < 0:
                if np.issubdtype(dtype, np.integer):
                    cpp.UtilityFunctions.push_warning("Unknown integer type %r, casting to int64" % dtype)
                    vartype = <int>cpp.PACKED_INT64_ARRAY
                elif np.issubdtype(dtype, np.str_):
                    vartype = <int>cpp.PACKED_STRING_ARRAY
                else:
                    vartype = <int>cpp.PACKED_FLOAT64_ARRAY
                    cpp.UtilityFunctions.push_warning("Unknown numeric type %r, casting to float64" % dtype)
        elif ndim == 2:
            ndim2_size = shape[2]
            if ndim2_size == 2:
                vartype = <int>cpp.PACKED_VECTOR2_ARRAY
            elif ndim2_size == 3:
                vartype = <int>cpp.PACKED_VECTOR2_ARRAY
            elif ndim2_size == 4:
                vartype = <int>cpp.PACKED_VECTOR4_ARRAY


cdef int pyobject_to_vartype(object p_obj) except -2:
    cdef int vartype = _pytype_to_vartype.get(type(p_obj), -1)
    cdef numpy.ndarray arr

    if vartype < 0:
        if PyLong_Check(p_obj):
            vartype = <int>cpp.INT
        elif PyFloat_Check(p_obj):
            vartype = <int>cpp.FLOAT
        elif PyUnicode_Check(p_obj) or PyBytes_Check(p_obj):
            vartype = <int>cpp.STRING
        # elif PyByteArray_Check(p_obj):
        #     vartype = <int>cpp.PACKED_BYTE_ARRAY
        elif PyArray_Check(p_obj):
            arr = p_obj
            vartype = array_to_vartype(arr)
        elif PyObject_CheckBuffer(p_obj):
            arr = np.array(p_obj)
            vartype = array_to_vartype(arr)
        elif PySequence_Check(p_obj):
            vartype = <int>cpp.ARRAY

    return vartype


cdef public void variant_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int vartype = pyobject_to_vartype(p_obj)
    cdef variant_type_from_pyobject_func_t func
    cdef str msg

    if vartype < 0:
        vartype = 0
        msg = "Unknown python object %r, could not convert, interpret as None"
        cpp.UtilityFunctions.push_warning(msg % p_obj)

    func = variant_from_pyobject_funcs[vartype]
    if func != NULL:
        func(p_obj, r_ret)
    else:
        msg = "NOT IMPLEMENTED: convertion of %r types from Python objects, interpret as None"
        cpp.UtilityFunctions.push_error(msg % variant_type_to_str(<cpp.VariantType>vartype))
        variant_nil_from_pyobject(None, r_ret)
