"""
Python representations of Godot Variant types

They cannot do much yet.
Used to hold converted Godot data and declare Godot types.
"""

from gdextension_interface cimport *
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from cpython cimport PyObject, ref
cimport godot_cpp as cpp

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy


import numpy as np
import _godot_type_tuples as tt


numpy._import_array()

__all__ = [
    'Nil',  # is None

    'bool',  # bool or TODO np.bool or ndarray as array(x, dtype=np.bool), shape = ()
    'int',  # int or TODO np.int64 or np.int32 or np.int8 or ndarray as array(x, dtype=intN), shape = ()
    'float',  # float or TODO np.float64 or np.float32 or ndarray as array(x, dtype=floatN), shape = ()
    'String',  # str or bytes

    'Vector2',  # subtype of ndarray as array([x, y], dtype=float32), shape = (2,)
    'Vector2i',  # subtype of ndarray as array([x, y], dtype=int32), shape = (2,)
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
    'AABB',  # TODO: subtype of ndarray as array([x, y, z, sx, sy, sz], dtype=float32), shape = (6,)
             # slices: Vector3, Vector3
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


cdef bint issubscriptable(object obj):
    return isinstance(obj, (np.ndarray, tuple, list)) or \
           (hasattr(obj, '__len__') and hasattr(obj, '__getitem__'))



cdef inline object PyArraySubType_NewFromBase(type subtype, numpy.ndarray base):
    cdef numpy.ndarray arr = numpy.PyArray_New(subtype, base.ndim, base.shape, numpy.PyArray_TYPE(base), NULL,
                                               base.data, 0, numpy.NPY_ARRAY_C_CONTIGUOUS | numpy.NPY_ARRAY_WRITEABLE,
                                               None)
    ref.Py_INCREF(base)
    numpy.PyArray_SetBaseObject(arr, base)

    return arr


ctypedef fused number_t:
    float
    double
    int8_t
    int16_t
    int32_t
    int64_t


Nil = None
bool = bool
int = int
float = float
String = str


cdef public object bool_to_pyobject(GDExtensionBool p_bool):
    return p_bool

cdef public object variant_bool_to_pyobject(const cpp.Variant &v):
    cdef bint ret = v.to_type[bint]()
    return ret

cdef public GDExtensionBool bool_from_pyobject(object p_obj):
    cdef object ret = bool(p_obj)
    return ret

cdef public cpp.Variant variant_bool_from_pyobject(object p_obj):
    cdef object _ret = bool(p_obj)
    cdef bint ret = _ret
    return cpp.Variant(ret)


cdef public object int_to_pyobject(int64_t p_int):
    return p_int

cdef public object variant_int_to_pyobject(const cpp.Variant &v):
    cdef int64_t ret = v.to_type[int64_t]()
    return ret

cdef public int64_t int_from_pyobject(object p_obj):
    cdef object ret = int(p_obj)
    return int(ret)

cdef public cpp.Variant variant_int_from_pyobject(object p_obj):
    cdef int64_t ret = int(p_obj)
    return cpp.Variant(ret)


cdef public object float_to_pyobject(double p_float):
    return p_float

cdef public object variant_float_to_pyobject(const cpp.Variant &v):
    cdef double ret = v.to_type[double]()
    return ret

cdef public double float_from_pyobject(object p_obj):
    cdef object ret = float(p_obj)
    return ret

cdef public cpp.Variant variant_float_from_pyobject(object p_obj):
    cdef object _ret = float(p_obj)
    cdef double ret = _ret
    return cpp.Variant(ret)


cdef public object string_to_pyobject(const cpp.String &p_string):
    return p_string.py_str()

cdef public object variant_string_to_pyobject(const cpp.Variant &v):
    cdef cpp.String ret = v.to_type[cpp.String]()
    return ret.py_str()

cdef public cpp.String string_from_pyobject(object p_obj):
    cdef object ret = p_obj.decode('utf-8') if isinstance(p_obj, bytes) else str(p_obj)
    return cpp.String(<const PyObject *>p_obj)

cdef public cpp.Variant variant_string_from_pyobject(object p_obj):
    cdef object _ret = p_obj.decode('utf-8') if isinstance(p_obj, bytes) else str(p_obj)
    cdef cpp.String ret = cpp.String(<const PyObject *>_ret)
    return cpp.Variant(ret)


class _Vector2Base(numpy.ndarray):
    # __array_priority__ = 10.0
    def __getattr__(self, str name):
        if name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'coord':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'coord':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))

class _Size2Base(numpy.ndarray):
    def __getattr__(self, str name):
        if name == 'width':
            return self[0]
        elif name == 'height':
            return self[1]
        elif name == 'coord':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'coord':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef inline numpy.ndarray array_from_vector2_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base

    # print("%r %r %r %r" % (subtype, dtype, args, kwargs))

    copy = kwargs.pop('copy', True)

    if kwargs:
        raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

    if args and len(args) == 2:
        base = np.array(args, dtype=dtype)
    elif args and len(args) == 1 and issubscriptable(args[0]) and len(args[0]) == 2:
        if isinstance(args[0], numpy.ndarray) and not copy:
            if args[0].dtype == dtype:
                base = args[0]
            else:
                base = args[0].as_type(dtype)
        else:
            base = np.array(args[0], dtype=dtype, copy=copy)
    elif len(args) == 0:
        base = np.array((0., 0.), dtype=dtype)
    else:
        raise TypeError("%r constructor accepts only one ('coordinates'), two ('x', 'y') or no arguments" % subtype)

    # print("Vector2 base: %r" % base)
    cdef numpy.ndarray ret = PyArraySubType_NewFromBase(subtype, base)

    return ret


class Vector2(_Vector2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if dtype not in (np.float32, np.float64, float):
            raise TypeError("%r accepts only 'float32' or 'float64' datatypes" % subtype)
        return array_from_vector2_args(subtype, dtype, args, kwargs)


class Size2(_Size2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if dtype not in (np.float32, np.float64, float):
            raise TypeError("%r accepts only 'float32' or 'float64' datatypes" % subtype)
        return array_from_vector2_args(subtype, dtype, args, kwargs)


class Vector2i(_Vector2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if dtype not in (np.int8, np.int16, np.int32, np.int64, np.int128, int):
            raise TypeError("%r accepts only 'intX' datatypes, got %r" % (subtype, dtype))
        return array_from_vector2_args(subtype, dtype, args, kwargs)


class Size2i(_Size2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if dtype not in (np.int8, np.int16, np.int32, np.int64, np.int128, int):
            raise TypeError("%r accepts only 'intX' datatypes, got %r" % (subtype, dtype))
        return array_from_vector2_args(subtype, dtype, args, kwargs)


cdef inline object array_from_carr_view(type arrtype, number_t [:] carr_view, copy=True):
    cdef numpy.ndarray pyarr = arrtype(carr_view, copy=copy)

    return pyarr


cdef public object vector2_to_pyobject(cpp.Vector2 &vec):
    cdef float [:] vec_view = vec.coord

    return array_from_carr_view[float](Vector2, vec_view)


cdef public object vector2i_to_pyobject(cpp.Vector2i &vec):
    cdef int32_t [:] vec_view = vec.coord

    return array_from_carr_view[int32_t](Vector2i, vec_view)


cdef public object variant_vector2_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector2 vec = v.to_type[cpp.Vector2]()
    cdef float [:] vec_view = vec.coord

    return array_from_carr_view[float](Vector2, vec_view)


cdef public object variant_vector2i_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector2i vec = v.to_type[cpp.Vector2i]()
    cdef int32_t [:] vec_view = vec.coord

    return array_from_carr_view[int32_t](Vector2i, vec_view)


cdef inline number_t [:] carr_view_from_pyobject(object obj, number_t [:] carr_view, dtype):
    cdef numpy.ndarray arr

    if not issubscriptable(obj) or not len(obj) == 2:
        cpp.UtilityFunctions.push_error("Cannot convert %r to godot-cpp %s" % (obj, obj.__class__.__name__))

        return carr_view

    if isinstance(obj, numpy.ndarray):
        arr = obj.as_type(dtype)
    else:
        arr = np.array(obj, dtype=dtype)

    cdef number_t [:] pyarr_view = arr
    carr_view = pyarr_view

    return carr_view


cdef public cpp.Vector2 vector2_from_pyobject(object obj):
    cdef cpp.Vector2 vec
    cdef float [:] carr_view = vec.coord
    carr_view = carr_view_from_pyobject[float](obj, carr_view, np.float32)

    return vec


cdef public cpp.Vector2i vector2i_from_pyobject(object obj):
    cdef cpp.Vector2i vec
    cdef int32_t [:] carr_view = vec.coord
    carr_view = carr_view_from_pyobject[int32_t](obj, carr_view, np.int32)

    return vec


cdef public cpp.Variant variant_vector2_from_pyobject(object obj):
    cdef cpp.Vector2 vec
    cdef float [:] carr_view = vec.coord
    carr_view = carr_view_from_pyobject[float](obj, carr_view, np.float32)

    return cpp.Variant(vec)


cdef public cpp.Variant variant_vector2i_from_pyobject(object obj):
    cdef cpp.Vector2i vec
    cdef int32_t [:] carr_view = vec.coord
    carr_view = carr_view_from_pyobject[int32_t](obj, carr_view, np.int32)

    return cpp.Variant(vec)


class _Rect2Base(numpy.ndarray):
    def __getattr__(self, str name):
        if name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'width':
            return self[2]
        elif name == 'height':
            return self[3]
        elif name == 'position':
            if self.dtype in (np.float32, np.float64, float):
                return Vector2(self[:2], dtype=self.dtype, copy=False)
            else:
                return Vector2i(self[:2], dtype=self.dtype, copy=False)
        elif name == 'size':
            if self.dtype in (np.float32, np.float64, float):
                return Size2(self[2:], dtype=self.dtype, copy=False)
            else:
                return Size2i(self[2:], dtype=self.dtype, copy=False)
        elif name == 'coord':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'width':
            self[2] = value
        elif name == 'height':
            self[3] = value
        elif name == 'position':
            self[:2] = value
        elif name == 'size':
            self[2:] = value
        elif name == 'coord':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef inline numpy.ndarray array_from_rect2_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base
    cdef object position, size

    copy = kwargs.pop('copy', False)

    if len(args) == 4:
        base = np.array(args, dtype=dtype, copy=copy)
    elif len(args) == 1:
        if isinstance(args[0], numpy.ndarray) and not copy:
            if args[0].dtype == dtype:
                base = args[0]
            else:
                base = args[0].as_type(dtype)
        else:
            base = np.array(args[0], dtype=dtype, copy=copy)
    else:
        size = args.pop() if len(args) > 1 else kwargs.pop('size', None)
        position = args.pop() if len(args) > 0 else kwargs.pop('position', None)

        if args:
            raise TypeError("Invalid positional argument %r" % args[0])
        elif kwargs:
            raise TypeError("Invalid keyword argument %r" % list(kwargs.keys).pop())

        if not issubscriptable(position) or len(position) != 2:
            raise TypeError("Invalid 'position' argument %r" % position)
        elif not issubscriptable(size) or len(size) != 2:
            raise TypeError("Invalid 'size' argument %r" % position)
        base = np.array([*position, *size], dtype=dtype, copy=copy)

    print("Rect2 base: %r" % base)

    cdef numpy.ndarray ret = PyArraySubType_NewFromBase(subtype, base)

    return ret


class Rect2(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if dtype not in (np.float32, np.float64, float):
            raise TypeError("%r accepts only 'float32' or 'float64' datatypes" % subtype)
        return array_from_rect2_args(subtype, dtype, args, kwargs)

class Rect2i(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if dtype not in (np.int8, np.int16, np.int32, np.int64, np.int128, int):
            raise TypeError("%r accepts only 'intX' datatypes, got %r" % (subtype, dtype))
        return array_from_rect2_args(subtype, dtype, args, kwargs)


cdef public class Vector3(numpy.ndarray) [object GDPyVector3, type GDPyVector3_Type]:
    pass


cdef public class Vector3i(numpy.ndarray) [object GDPyVector3i, type GDPyVector3i_Type]:
    pass


class Transform2D(np.matrix):
    pass

cdef public type GDPyTransform2D_TypePtr = Transform2D


cdef public class Vector4(numpy.ndarray) [object GDPyVector4, type GDPyVector4_Type]:
    pass


cdef public class Vector4i(numpy.ndarray) [object GDPyVector4i, type GDPyVector4i_Type]:
    pass


Plane = tt.Plane


cdef public class Quaternion(numpy.ndarray) [object GDPyQuaternion, type GDPyQuaternion_Type]:
    pass


AABB = tt.AABB


class Basis(np.matrix):
    pass

cdef public type GDPyBasis_TypePtr = Basis


Transform3D = tt.Transform3D


class Projection(np.matrix):
    pass

cdef public type GDPyProjection_TypePtr = Projection


cdef public class Color(numpy.ndarray) [object GDPyColor, type GDPyColor_Type]:
    pass


class StringName(str):
    pass


class NodePath(str):
    pass


class RID(int):
    pass


# Object, Callable, Signal are in gdextension module


Dictionary = dict
Array = list

PackedByteArray = bytearray


cdef public class PackedInt32Array(numpy.ndarray) [object GDPyPackedInt32Array, type GDPyPackedInt32Array_Type]:
    pass


cdef public class PackedInt64Array(numpy.ndarray) [object GDPyPackedInt64Array, type GDPyPackedInt64Array_Type]:
    pass


cdef public class PackedFloat32Array(numpy.ndarray) [object GDPyPackedFloat32Array, type GDPyPackedFloat32Array_Type]:
    pass


cdef public class PackedFloat64Array(numpy.ndarray) [object GDPyPackedFloat64Array, type GDPyPackedFloat64Array_Type]:
    pass


class PackedStringArray(tuple):
    pass


cdef public class PackedVector2Array(numpy.ndarray) [object GDPyPackedVector2Array, type GDPyPackedVector2Array_Type]:
    pass


cdef public class PackedVector3Array(numpy.ndarray) [object GDPyPackedVector3Array, type GDPyPackedVector3Array_Type]:
    pass


cdef public class PackedColorArray(numpy.ndarray) [object GDPyPackedColorArray, type GDPyPackedColorArray_Type]:
    pass


cdef public class PackedVector4Array(numpy.ndarray) [object GDPyPackedVector4Array, type GDPyPackedVector4Array_Type]:
    pass
