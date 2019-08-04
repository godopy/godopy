from godot_headers.gdnative_api cimport godot_object

from .cpp.core_types cimport *

# cdef extern from * namespace "godot" nogil:
#     cdef cppclass AABB
#     cdef cppclass Array
#     cdef cppclass Color
#     cdef cppclass Dictionary
#     cdef cppclass NodePath
#     cdef cppclass Plane
#     cdef cppclass PoolByteArray
#     cdef cppclass PoolIntArray
#     cdef cppclass PoolRealArray
#     cdef cppclass PoolStringArray
#     cdef cppclass PoolVector2Array
#     cdef cppclass PoolVector3Array
#     cdef cppclass PoolColorArray
#     cdef cppclass Quat
#     cdef cppclass Rect2
#     cdef cppclass RID
#     cdef cppclass CharString
#     cdef cppclass String
#     cdef cppclass Transform
#     cdef cppclass Transform2D
#     cdef cppclass Vector2
#     cdef cppclass Vector3

include "core/defs.pxi"

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
    cdef AABB _cpp_object
    @staticmethod
    cdef GodotAABB from_cpp(AABB _cpp_object)

cdef class GodotArrayBase(GodotCoreTypeWrapper):
    pass

cdef class GodotArray(GodotArrayBase):
    cdef Array _cpp_object
    @staticmethod
    cdef GodotArray from_cpp(Array _cpp_object)

cdef class GodotBasis(GodotCoreTypeWrapper):
    cdef Basis _cpp_object
    @staticmethod
    cdef GodotBasis from_cpp(Basis _cpp_object)

cdef class GodotColor(GodotCoreTypeWrapper):
    cdef Color _cpp_object
    @staticmethod
    cdef GodotColor from_cpp(Color _cpp_object)

cdef class GodotDictionary(GodotCoreTypeWrapper):
    cdef Dictionary _cpp_object
    @staticmethod
    cdef GodotDictionary from_cpp(Dictionary _cpp_object)

cdef class GodotNodePath(GodotCoreTypeWrapper):
    cdef NodePath _cpp_object
    @staticmethod
    cdef GodotNodePath from_cpp(NodePath _cpp_object)

cdef class GodotPlane(GodotCoreTypeWrapper):
    cdef Plane _cpp_object
    @staticmethod
    cdef GodotPlane from_cpp(Plane _cpp_object)

cdef class GodotPoolArrayBase(GodotArrayBase):
    pass

cdef class GodotPoolByteArray(GodotPoolArrayBase):
    cdef PoolByteArray _cpp_object
    @staticmethod
    cdef GodotPoolByteArray from_cpp(PoolByteArray _cpp_object)

cdef class GodotPoolIntArray(GodotPoolArrayBase):
    cdef PoolIntArray _cpp_object
    @staticmethod
    cdef GodotPoolIntArray from_cpp(PoolIntArray _cpp_object)

cdef class GodotPoolRealArray(GodotPoolArrayBase):
    cdef PoolRealArray _cpp_object
    @staticmethod
    cdef GodotPoolRealArray from_cpp(PoolRealArray _cpp_object)

cdef class GodotPoolStringArray(GodotPoolArrayBase):
    cdef PoolStringArray _cpp_object
    @staticmethod
    cdef GodotPoolStringArray from_cpp(PoolStringArray _cpp_object)

cdef class GodotPoolVector2Array(GodotPoolArrayBase):
    cdef PoolVector2Array _cpp_object
    @staticmethod
    cdef GodotPoolVector2Array from_cpp(PoolVector2Array _cpp_object)

cdef class GodotPoolVector3Array(GodotPoolArrayBase):
    cdef PoolVector3Array _cpp_object
    @staticmethod
    cdef GodotPoolVector3Array from_cpp(PoolVector3Array _cpp_object)

cdef class GodotPoolColorArray(GodotPoolArrayBase):
    cdef PoolColorArray _cpp_object
    @staticmethod
    cdef GodotPoolColorArray from_cpp(PoolColorArray _cpp_object)

cdef class GodotQuat(GodotCoreTypeWrapper):
    cdef Quat _cpp_object
    @staticmethod
    cdef GodotQuat from_cpp(Quat _cpp_object)

cdef class GodotRect2(GodotCoreTypeWrapper):
    cdef Rect2 _cpp_object
    @staticmethod
    cdef GodotRect2 from_cpp(Rect2 _cpp_object)

cdef class GodotRID(GodotCoreTypeWrapper):
    cdef RID _cpp_object
    @staticmethod
    cdef GodotRID from_cpp(RID _cpp_object)

cdef class GodotCharString(GodotCoreTypeWrapper):
    cdef CharString _cpp_object
    @staticmethod
    cdef GodotCharString from_cpp(CharString _cpp_object)

cdef class GodotString(GodotCoreTypeWrapper):
    cdef String _cpp_object
    @staticmethod
    cdef GodotString from_cpp(String _cpp_object)

cdef class GodotTransform(GodotCoreTypeWrapper):
    cdef Transform _cpp_object
    @staticmethod
    cdef GodotTransform from_cpp(Transform _cpp_object)

cdef class GodotTransform2D(GodotCoreTypeWrapper):
    cdef Transform2D _cpp_object
    @staticmethod
    cdef GodotTransform2D from_cpp(Transform2D _cpp_object)

cdef class GodotVector2(GodotCoreTypeWrapper):
    cdef Vector2 _cpp_object
    @staticmethod
    cdef GodotVector2 from_cpp(Vector2 _cpp_object)

cdef class GodotVector3(GodotCoreTypeWrapper):
    cdef Vector3 _cpp_object
    @staticmethod
    cdef GodotVector3 from_cpp(Vector3 _cpp_object)

cdef class _Wrapped:
    cdef godot_object *_owner
    cdef bint ___CLASS_IS_SCRIPT
    cdef bint ___CLASS_IS_SINGLETON
    cdef int ___CLASS_BINDING_LEVEL

cdef class _PyWrapped(_Wrapped):
    pass


include "core/signal_arguments.pxi"

cdef dict CythonTagDB
cdef dict PythonTagDB
cdef dict __instance_map

cdef register_cython_type(type cls)
cdef register_python_type(type cls)
cdef register_global_cython_type(type cls, str api_name)
cdef register_global_python_type(type cls, str api_name)

cdef get_instance_from_owner(godot_object *instance)
