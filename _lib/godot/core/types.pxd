from . cimport cpp_types as cpp
from godot_headers cimport gdnative_api as gdnative

from ._wrapped cimport _Wrapped

from godopy cimport numpy as np

cdef class CoreTypeWrapper:
    cdef bint _initialized

    # cdef inline int _internal_check(self) except -1
    # cdef inline ValueError _init_value_error(self, object value)
    cdef inline int _internal_check(self) except -1:
        if not self._initialized:
            raise RuntimeError('%r insance was not initialized properly')

    cdef inline object _init_value_error(self, object value):
        return ValueError('Invalid %r init value: %r' % (self.__class__, value))

    cdef inline object _argument_error(self, object value):
        return ValueError('Invalid argument: %r' %  value)


cdef class AABB(CoreTypeWrapper):
    cdef cpp.AABB _cpp_object

    @staticmethod
    cdef AABB from_cpp(cpp.AABB _cpp_object)

    cdef cpp.AABB to_cpp(self)


cdef class Array(CoreTypeWrapper):
    cdef cpp.Array _cpp_object

    @staticmethod
    cdef Array from_cpp(cpp.Array _cpp_object)

    cdef cpp.Array to_cpp(self)


cdef class Basis(CoreTypeWrapper):
    cdef cpp.Basis _cpp_object

    @staticmethod
    cdef Basis from_cpp(cpp.Basis _cpp_object)

    cdef cpp.Basis to_cpp(self)


cdef class Color(CoreTypeWrapper):
    cdef cpp.Color _cpp_object

    @staticmethod
    cdef Color from_cpp(cpp.Color _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.Color _cpp_object)

    cdef cpp.Color to_cpp(self)


cdef class Dictionary(CoreTypeWrapper):
    cdef cpp.Dictionary _cpp_object

    @staticmethod
    cdef Dictionary from_cpp(cpp.Dictionary _cpp_object)

    cdef cpp.Dictionary to_cpp(self)


cdef class NodePath(CoreTypeWrapper):
    cdef cpp.NodePath _cpp_object

    @staticmethod
    cdef NodePath from_cpp(cpp.NodePath _cpp_object)

    cdef cpp.NodePath to_cpp(self)


cdef class Plane(CoreTypeWrapper):
    cdef cpp.Plane _cpp_object

    @staticmethod
    cdef Plane from_cpp(cpp.Plane _cpp_object)

    cdef cpp.Plane to_cpp(self)


cdef class PoolArray(CoreTypeWrapper):
    pass


cdef class PoolArrayReadAccess:
    cdef gdnative.godot_pool_array_read_access *_read_access
    cdef np.npy_intp _size

cdef class PoolArrayWriteAccess:
    cdef gdnative.godot_pool_array_write_access *_write_access
    cdef np.npy_intp _size


cdef class PoolByteArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolByteArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolByteArray(PoolArray):
    cdef cpp.PoolByteArray _cpp_object

    @staticmethod
    cdef PoolByteArray from_cpp(cpp.PoolByteArray _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolByteArray _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolByteArray _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolByteArray _cpp_object)

    cdef cpp.PoolByteArray to_cpp(self)


cdef class PoolIntArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolIntArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolIntArray(PoolArray):
    cdef cpp.PoolIntArray _cpp_object

    @staticmethod
    cdef PoolIntArray from_cpp(cpp.PoolIntArray _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolIntArray _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolIntArray _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolIntArray _cpp_object)

    cdef cpp.PoolIntArray to_cpp(self)


cdef class PoolRealArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolRealArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolRealArray(PoolArray):
    cdef cpp.PoolRealArray _cpp_object

    @staticmethod
    cdef PoolRealArray from_cpp(cpp.PoolRealArray _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolRealArray _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolRealArray _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolRealArray _cpp_object)

    cdef cpp.PoolRealArray to_cpp(self)


cdef class PoolStringArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolStringArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolStringArray(PoolArray):
    cdef cpp.PoolStringArray _cpp_object

    @staticmethod
    cdef PoolStringArray from_cpp(cpp.PoolStringArray _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolStringArray _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolStringArray _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolStringArray _cpp_object)

    cdef cpp.PoolStringArray to_cpp(self)


cdef class PoolVector2ArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolVector2ArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolVector2Array(PoolArray):
    cdef cpp.PoolVector2Array _cpp_object

    @staticmethod
    cdef PoolVector2Array from_cpp(cpp.PoolVector2Array _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolVector2Array _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolVector2Array _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolVector2Array _cpp_object)

    cdef cpp.PoolVector2Array to_cpp(self)


cdef class PoolVector3ArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolVector3ArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolVector3Array(PoolArray):
    cdef cpp.PoolVector3Array _cpp_object

    @staticmethod
    cdef PoolVector3Array from_cpp(cpp.PoolVector3Array _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolVector3Array _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolVector3Array _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolVector3Array _cpp_object)

    cdef cpp.PoolVector3Array to_cpp(self)


cdef class PoolColorArrayReadAccess(PoolArrayReadAccess):
    pass

cdef class PoolColorArrayWriteAccess(PoolArrayWriteAccess):
    pass

cdef class PoolColorArray(PoolArray):
    cdef cpp.PoolColorArray _cpp_object

    @staticmethod
    cdef PoolColorArray from_cpp(cpp.PoolColorArray _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolColorArray _cpp_object, writable=*)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolColorArray _cpp_object)

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolColorArray _cpp_object)

    cdef cpp.PoolColorArray to_cpp(self)


cdef class Quat(CoreTypeWrapper):
    cdef cpp.Quat _cpp_object

    @staticmethod
    cdef Quat from_cpp(cpp.Quat _cpp_object)

    cdef cpp.Quat to_cpp(self)


cdef class Rect2(CoreTypeWrapper):
    cdef cpp.Rect2 _cpp_object

    @staticmethod
    cdef Rect2 from_cpp(cpp.Rect2 _cpp_object)

    cdef cpp.Rect2 to_cpp(self)


cdef class RID(CoreTypeWrapper):
    cdef gdnative.godot_rid _godot_rid

    @staticmethod
    cdef RID from_cpp(cpp.RID _cpp_object)

    @staticmethod
    cdef RID from_godot_object(gdnative.godot_object *_godot_object)

    cdef cpp.RID to_cpp(self)
    cdef gdnative.godot_rid *to_godot_rid(self)


cdef class CharString(CoreTypeWrapper):
    cdef cpp.CharString _cpp_object

    @staticmethod
    cdef CharString from_cpp(cpp.CharString _cpp_object)

    cdef cpp.CharString to_cpp(self)


cdef class String(CoreTypeWrapper):
    cdef cpp.String _cpp_object

    @staticmethod
    cdef String from_cpp(cpp.String _cpp_object)

    cdef cpp.String to_cpp(self)


cdef class Transform(CoreTypeWrapper):
    cdef cpp.Transform _cpp_object

    @staticmethod
    cdef Transform from_cpp(cpp.Transform _cpp_object)

    cdef cpp.Transform to_cpp(self)


cdef class Transform2D(CoreTypeWrapper):
    cdef cpp.Transform2D _cpp_object

    @staticmethod
    cdef Transform2D from_cpp(cpp.Transform2D _cpp_object)

    cdef cpp.Transform2D to_cpp(self)


cdef class Vector2(CoreTypeWrapper):
    cdef cpp.Vector2 _cpp_object

    @staticmethod
    cdef Vector2 from_cpp(cpp.Vector2 _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.Vector2 _cpp_object)

    cdef cpp.Vector2 to_cpp(self)


cdef class Point2(Vector2):
    @staticmethod
    cdef Point2 from_cpp(cpp.Point2 _cpp_object)


cdef class Size2(Vector2):
    @staticmethod
    cdef Size2 from_cpp(cpp.Size2 _cpp_object)


cdef class Vector3(CoreTypeWrapper):
    cdef cpp.Vector3 _cpp_object

    @staticmethod
    cdef Vector3 from_cpp(cpp.Vector3 _cpp_object)

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.Vector3 _cpp_object)

    cdef cpp.Vector3 to_cpp(self)
