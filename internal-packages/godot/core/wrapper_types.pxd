from . cimport cpp_types as cpp

cdef class CoreTypeWrapper:
    cdef bint _initialized

    # cdef inline int _internal_check(self) except -1
    # cdef inline ValueError _init_value_error(self, object value)
    cdef inline int _internal_check(self) except -1:
        if not self._initialized:
            raise RuntimeError('%r insance was not initialized properly')

    cdef inline object _init_value_error(self, object value):
        return ValueError('Bad %r init value: %r' % (self.__class__, value))

cdef class AABB(CoreTypeWrapper):
    cdef cpp.AABB _cpp_object
    @staticmethod
    cdef AABB from_cpp(cpp.AABB _cpp_object)

cdef class ArrayBase(CoreTypeWrapper):
    pass

cdef class Array(ArrayBase):
    cdef cpp.Array _cpp_object
    @staticmethod
    cdef Array from_cpp(cpp.Array _cpp_object)

cdef class Basis(CoreTypeWrapper):
    cdef cpp.Basis _cpp_object
    @staticmethod
    cdef Basis from_cpp(cpp.Basis _cpp_object)

cdef class Color(CoreTypeWrapper):
    cdef cpp.Color _cpp_object
    @staticmethod
    cdef Color from_cpp(cpp.Color _cpp_object)

cdef class Dictionary(CoreTypeWrapper):
    cdef cpp.Dictionary _cpp_object
    @staticmethod
    cdef Dictionary from_cpp(cpp.Dictionary _cpp_object)

cdef class NodePath(CoreTypeWrapper):
    cdef cpp.NodePath _cpp_object
    @staticmethod
    cdef NodePath from_cpp(cpp.NodePath _cpp_object)

cdef class Plane(CoreTypeWrapper):
    cdef cpp.Plane _cpp_object
    @staticmethod
    cdef Plane from_cpp(cpp.Plane _cpp_object)

cdef class PoolArrayBase(ArrayBase):
    pass

cdef class PoolByteArray(PoolArrayBase):
    cdef cpp.PoolByteArray _cpp_object
    @staticmethod
    cdef PoolByteArray from_cpp(cpp.PoolByteArray _cpp_object)

cdef class PoolIntArray(PoolArrayBase):
    cdef cpp.PoolIntArray _cpp_object
    @staticmethod
    cdef PoolIntArray from_cpp(cpp.PoolIntArray _cpp_object)

cdef class PoolRealArray(PoolArrayBase):
    cdef cpp.PoolRealArray _cpp_object
    @staticmethod
    cdef PoolRealArray from_cpp(cpp.PoolRealArray _cpp_object)

cdef class PoolStringArray(PoolArrayBase):
    cdef cpp.PoolStringArray _cpp_object
    @staticmethod
    cdef PoolStringArray from_cpp(cpp.PoolStringArray _cpp_object)

cdef class PoolVector2Array(PoolArrayBase):
    cdef cpp.PoolVector2Array _cpp_object
    @staticmethod
    cdef PoolVector2Array from_cpp(cpp.PoolVector2Array _cpp_object)

cdef class PoolVector3Array(PoolArrayBase):
    cdef cpp.PoolVector3Array _cpp_object
    @staticmethod
    cdef PoolVector3Array from_cpp(cpp.PoolVector3Array _cpp_object)

cdef class PoolColorArray(PoolArrayBase):
    cdef cpp.PoolColorArray _cpp_object
    @staticmethod
    cdef PoolColorArray from_cpp(cpp.PoolColorArray _cpp_object)

cdef class Quat(CoreTypeWrapper):
    cdef cpp.Quat _cpp_object
    @staticmethod
    cdef Quat from_cpp(cpp.Quat _cpp_object)

cdef class Rect2(CoreTypeWrapper):
    cdef cpp.Rect2 _cpp_object
    @staticmethod
    cdef Rect2 from_cpp(cpp.Rect2 _cpp_object)

cdef class RID(CoreTypeWrapper):
    cdef cpp.RID _cpp_object
    @staticmethod
    cdef RID from_cpp(cpp.RID _cpp_object)

cdef class CharString(CoreTypeWrapper):
    cdef cpp.CharString _cpp_object
    @staticmethod
    cdef CharString from_cpp(cpp.CharString _cpp_object)

cdef class String(CoreTypeWrapper):
    cdef cpp.String _cpp_object
    @staticmethod
    cdef String from_cpp(cpp.String _cpp_object)

cdef class Transform(CoreTypeWrapper):
    cdef cpp.Transform _cpp_object
    @staticmethod
    cdef Transform from_cpp(cpp.Transform _cpp_object)

cdef class Transform2D(CoreTypeWrapper):
    cdef cpp.Transform2D _cpp_object
    @staticmethod
    cdef Transform2D from_cpp(cpp.Transform2D _cpp_object)

cdef class Vector2(CoreTypeWrapper):
    cdef cpp.Vector2 _cpp_object
    @staticmethod
    cdef Vector2 from_cpp(cpp.Vector2 _cpp_object)

cdef class Vector3(CoreTypeWrapper):
    cdef cpp.Vector3 _cpp_object
    @staticmethod
    cdef Vector3 from_cpp(cpp.Vector3 _cpp_object)



