"""
Python versions of Godot Variant types
"""

from gdextension_interface cimport *
from binding cimport *
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stddef cimport wchar_t
from cpython cimport (
    PyObject, ref, PyUnicode_AsWideCharString, PyUnicode_FromWideChar,
    PyBool_Check, PyLong_Check, PyFloat_Check, PyUnicode_Check, PyBytes_Check,
    PyObject_IsTrue
)
cimport godot_cpp as cpp

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy
from numpy cimport PyArray_New, PyArray_Check, PyArray_TYPE, NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_WRITEABLE


import numpy as np
import _godot_type_tuples as tt


numpy._import_array()

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


ctypedef fused number_t:
    float
    double
    int8_t
    int16_t
    int32_t
    int64_t


cdef bint issubscriptable(object obj):
    return isinstance(obj, (np.ndarray, tuple, list)) or \
           (hasattr(obj, '__len__') and hasattr(obj, '__getitem__'))


cdef inline object PyArraySubType_NewFromBase(type subtype, numpy.ndarray base):
    cdef numpy.ndarray arr = PyArray_New(subtype, base.ndim, base.shape, PyArray_TYPE(base), NULL,
                                         base.data, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)
    ref.Py_INCREF(base)
    numpy.PyArray_SetBaseObject(arr, base)

    return arr


cdef inline object array_from_carr_view(type arrtype, number_t [:] carr_view, copy=True):
    cdef numpy.ndarray pyarr = arrtype(carr_view, copy=copy)

    return pyarr


cdef inline number_t [:] carr_view_from_pyobject(object obj, number_t [:] carr_view, dtype):
    cdef numpy.ndarray arr

    if not issubscriptable(obj) or not len(obj) == 2:
        cpp.UtilityFunctions.push_error("Cannot convert %r to godot-cpp %s" % (obj, obj.__class__.__name__))

        return carr_view

    if isinstance(obj, numpy.ndarray):
        if obj.dtype == dtype:
            arr = obj
        else:
            cpp.UtilityFunctions.push_warning("Cast from %r to %r during Godot math type convertion" % (obj.dtype, dtype))
            arr = obj.astype(dtype)
    else:
        arr = np.array(obj, dtype=dtype)

    cdef number_t [:] pyarr_view = arr
    carr_view[:] = pyarr_view

    return carr_view


include "godot_types_includes/atomic.pxi"
include "godot_types_includes/vector2.pxi"
include "godot_types_includes/rect2.pxi"
include "godot_types_includes/vector3.pxi"
include "godot_types_includes/transform2d.pxi"
include "godot_types_includes/vector4.pxi"
include "godot_types_includes/plane.pxi"
include "godot_types_includes/quaternion.pxi"
include "godot_types_includes/aabb.pxi"
include "godot_types_includes/basis.pxi"
include "godot_types_includes/transform3d.pxi"
include "godot_types_includes/projection.pxi"
include "godot_types_includes/color.pxi"
include "godot_types_includes/misc.pxi"
include "godot_types_includes/packed_1dim_array.pxi"
include "godot_types_includes/packed_2dim_array.pxi"
