from . cimport cpp_types as cpp

cdef class GodotCoreTypeWrapper:
    cdef bint _initialized

    # cdef inline int _internal_check(self) except -1
    # cdef inline ValueError _init_value_error(self, object value)
    cdef inline int _internal_check(self) except -1:
        if not self._initialized:
            raise RuntimeError('%r insance was not initialized properly')

    cdef inline object _init_value_error(self, object value):
        return ValueError('Bad %r init value: %r' % (self.__class__, value))

cdef class GodotAABB(GodotCoreTypeWrapper):
    cdef cpp.AABB _cpp_object
    @staticmethod
    cdef GodotAABB from_cpp(cpp.AABB _cpp_object)

cdef class GodotArrayBase(GodotCoreTypeWrapper):
    pass

cdef class GodotArray(GodotArrayBase):
    cdef cpp.Array _cpp_object
    @staticmethod
    cdef GodotArray from_cpp(cpp.Array _cpp_object)

cdef class GodotBasis(GodotCoreTypeWrapper):
    cdef cpp.Basis _cpp_object
    @staticmethod
    cdef GodotBasis from_cpp(cpp.Basis _cpp_object)

cdef class GodotColor(GodotCoreTypeWrapper):
    cdef cpp.Color _cpp_object
    @staticmethod
    cdef GodotColor from_cpp(cpp.Color _cpp_object)

cdef class GodotDictionary(GodotCoreTypeWrapper):
    cdef cpp.Dictionary _cpp_object
    @staticmethod
    cdef GodotDictionary from_cpp(cpp.Dictionary _cpp_object)

cdef class GodotNodePath(GodotCoreTypeWrapper):
    cdef cpp.NodePath _cpp_object
    @staticmethod
    cdef GodotNodePath from_cpp(cpp.NodePath _cpp_object)

cdef class GodotPlane(GodotCoreTypeWrapper):
    cdef cpp.Plane _cpp_object
    @staticmethod
    cdef GodotPlane from_cpp(cpp.Plane _cpp_object)

cdef class GodotPoolArrayBase(GodotArrayBase):
    pass

cdef class GodotPoolByteArray(GodotPoolArrayBase):
    cdef cpp.PoolByteArray _cpp_object
    @staticmethod
    cdef GodotPoolByteArray from_cpp(cpp.PoolByteArray _cpp_object)

cdef class GodotPoolIntArray(GodotPoolArrayBase):
    cdef cpp.PoolIntArray _cpp_object
    @staticmethod
    cdef GodotPoolIntArray from_cpp(cpp.PoolIntArray _cpp_object)

cdef class GodotPoolRealArray(GodotPoolArrayBase):
    cdef cpp.PoolRealArray _cpp_object
    @staticmethod
    cdef GodotPoolRealArray from_cpp(cpp.PoolRealArray _cpp_object)

cdef class GodotPoolStringArray(GodotPoolArrayBase):
    cdef cpp.PoolStringArray _cpp_object
    @staticmethod
    cdef GodotPoolStringArray from_cpp(cpp.PoolStringArray _cpp_object)

cdef class GodotPoolVector2Array(GodotPoolArrayBase):
    cdef cpp.PoolVector2Array _cpp_object
    @staticmethod
    cdef GodotPoolVector2Array from_cpp(cpp.PoolVector2Array _cpp_object)

cdef class GodotPoolVector3Array(GodotPoolArrayBase):
    cdef cpp.PoolVector3Array _cpp_object
    @staticmethod
    cdef GodotPoolVector3Array from_cpp(cpp.PoolVector3Array _cpp_object)

cdef class GodotPoolColorArray(GodotPoolArrayBase):
    cdef cpp.PoolColorArray _cpp_object
    @staticmethod
    cdef GodotPoolColorArray from_cpp(cpp.PoolColorArray _cpp_object)

cdef class GodotQuat(GodotCoreTypeWrapper):
    cdef cpp.Quat _cpp_object
    @staticmethod
    cdef GodotQuat from_cpp(cpp.Quat _cpp_object)

cdef class GodotRect2(GodotCoreTypeWrapper):
    cdef cpp.Rect2 _cpp_object
    @staticmethod
    cdef GodotRect2 from_cpp(cpp.Rect2 _cpp_object)

cdef class GodotRID(GodotCoreTypeWrapper):
    cdef cpp.RID _cpp_object
    @staticmethod
    cdef GodotRID from_cpp(cpp.RID _cpp_object)

cdef class GodotCharString(GodotCoreTypeWrapper):
    cdef cpp.CharString _cpp_object
    @staticmethod
    cdef GodotCharString from_cpp(cpp.CharString _cpp_object)

cdef class GodotString(GodotCoreTypeWrapper):
    cdef cpp.String _cpp_object
    @staticmethod
    cdef GodotString from_cpp(cpp.String _cpp_object)

cdef class GodotTransform(GodotCoreTypeWrapper):
    cdef cpp.Transform _cpp_object
    @staticmethod
    cdef GodotTransform from_cpp(cpp.Transform _cpp_object)

cdef class GodotTransform2D(GodotCoreTypeWrapper):
    cdef cpp.Transform2D _cpp_object
    @staticmethod
    cdef GodotTransform2D from_cpp(cpp.Transform2D _cpp_object)

cdef class GodotVector2(GodotCoreTypeWrapper):
    cdef cpp.Vector2 _cpp_object
    @staticmethod
    cdef GodotVector2 from_cpp(cpp.Vector2 _cpp_object)

cdef class GodotVector3(GodotCoreTypeWrapper):
    cdef cpp.Vector3 _cpp_object
    @staticmethod
    cdef GodotVector3 from_cpp(cpp.Vector3 _cpp_object)



