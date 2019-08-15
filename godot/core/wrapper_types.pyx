from libc.stdint cimport uint64_t, uint32_t, uint8_t, int64_t, int32_t
from libc.stddef cimport wchar_t

from godot_headers.gdnative_api cimport godot_object, godot_array, godot_vector2

from ._wrapped cimport _Wrapped, _PyWrapped
from . cimport cpp_types as cpp


def is_godot_wrapper_instance(object obj, object instances):
    return isinstance(obj, instances) and (<GodotCoreTypeWrapper?>obj).__ready

cdef class GodotCoreTypeWrapper:
    def __cinit__(self):
        self._initialized = False

    # cdef inline int _internal_check(self) except -1:
    #     if not self._initialized:
    #         raise RuntimeError('%r insance was not initialized properly')

    # cdef inline ValueError _init_value_error(self, object value):
    #     return ValueError('Bad %r init value: %r' % (self.__class__, value))


cdef class GodotAABB(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotAABB from_cpp(cpp.AABB _cpp_object):
        cdef GodotAABB self = GodotAABB.__new__(GodotAABB)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object pos=None, object size=None):
        if pos is not None and not is_godot_wrapper_instance(pos, GodotVector3):
           raise self._init_value_error(pos)
        if size is not None and not is_godot_wrapper_instance(pos, GodotVector3):
            raise self._init_value_error(size)

        if pos is not None and size is not None:
            self._cpp_object = cpp.AABB((<GodotVector3>pos)._cpp_object, (<GodotVector3>size)._cpp_object)
        else:
            self._cpp_object = cpp.AABB()
            if pos is not None:
                self._cpp_object.set_position((<GodotVector3>pos)._cpp_object)
            if size is not None:
                self._cpp_object.set_size((<GodotVector3>size)._cpp_object)

        self._initialized = True


cdef class GodotArrayBase(GodotCoreTypeWrapper):
    pass


cdef class GodotArray(GodotArrayBase):
    @staticmethod
    cdef GodotArray from_cpp(cpp.Array _cpp_object):
        cdef GodotArray self = GodotArray.__new__(GodotArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        if other is None:
            self._cpp_object = cpp.Array()
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.Array((<GodotArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolByteArray):
            self._cpp_object = cpp.Array((<GodotPoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolIntArray):
            self._cpp_object = cpp.Array((<GodotPoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolRealArray):
            self._cpp_object = cpp.Array((<GodotPoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolStringArray):
            self._cpp_object = cpp.Array((<GodotPoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolVector2Array):
            self._cpp_object = cpp.Array((<GodotPoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolVector3Array):
            self._cpp_object = cpp.Array((<GodotPoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolColorArray):
            self._cpp_object = cpp.Array((<GodotPoolByteArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


    @staticmethod
    def make(*values):
        array = GodotArray()

        for value in values:
            array._cpp_object.append(<const cpp.Variant &>value)

        return array

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return <object>self._cpp_object[<const int>item]

    # def __setitem__(self, int item, object value):
    #     self._internal_check()
    #     self._cpp_object[<const int>item] = <const cpp.Variant &>value


cdef class GodotBasis(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotBasis from_cpp(cpp.Basis _cpp_object):
        cdef GodotBasis self = GodotBasis.__new__(GodotBasis)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        self._cpp_object = cpp.Basis()
        self._initialized = True


cdef class GodotColor(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotColor from_cpp(cpp.Color _cpp_object):
        cdef GodotColor self = GodotColor.__new__(GodotColor)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float r=0, float g=0, float b=0, float a=1):
        self._cpp_object = cpp.Color(r, g, b, a)
        self._initialized = True

    @staticmethod
    def hex(self, uint32_t value):
        return GodotColor.from_cpp(cpp.Color.hex(value))

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return self._cpp_object[item]

    # def __setitem__(self, int item, float value):
    #     self._internal_check()
    #     self._cpp_object[item] = value


cdef class GodotDictionary(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotDictionary from_cpp(cpp.Dictionary _cpp_object):
        cdef GodotDictionary self = GodotDictionary.__new__(GodotDictionary)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Dictionary()
        self._initialized = True


cdef class GodotNodePath(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotNodePath from_cpp(cpp.NodePath _cpp_object):
        cdef GodotNodePath self = GodotNodePath.__new__(GodotNodePath)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.NodePath()
        self._initialized = True


cdef class GodotPlane(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotPlane from_cpp(cpp.Plane _cpp_object):
        cdef GodotPlane self = GodotPlane.__new__(GodotPlane)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Plane()
        self._initialized = True


cdef class GodotPoolArrayBase(GodotArrayBase):
    pass


cdef class GodotPoolByteArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolByteArray from_cpp(cpp.PoolByteArray _cpp_object):
        cdef GodotPoolByteArray self = GodotPoolByteArray.__new__(GodotPoolByteArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolByteArray()
        elif is_godot_wrapper_instance(other, GodotPoolByteArray):
            self._cpp_object = cpp.PoolByteArray((<GodotPoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolByteArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolIntArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolIntArray from_cpp(cpp.PoolIntArray _cpp_object):
        cdef GodotPoolIntArray self = GodotPoolIntArray.__new__(GodotPoolIntArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolIntArray()
        elif is_godot_wrapper_instance(other, GodotPoolIntArray):
            self._cpp_object = cpp.PoolIntArray((<GodotPoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolIntArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolRealArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolRealArray from_cpp(cpp.PoolRealArray _cpp_object):
        cdef GodotPoolRealArray self = GodotPoolRealArray.__new__(GodotPoolRealArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolRealArray()
        elif is_godot_wrapper_instance(other, GodotPoolRealArray):
            self._cpp_object = cpp.PoolRealArray((<GodotPoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolRealArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)
        self._initialized = True


cdef class GodotPoolStringArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolStringArray from_cpp(cpp.PoolStringArray _cpp_object):
        cdef GodotPoolStringArray self = GodotPoolStringArray.__new__(GodotPoolStringArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolStringArray()
        elif is_godot_wrapper_instance(other, GodotPoolStringArray):
            self._cpp_object = cpp.PoolStringArray((<GodotPoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolStringArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolVector2Array(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolVector2Array from_cpp(cpp.PoolVector2Array _cpp_object):
        cdef GodotPoolVector2Array self = GodotPoolVector2Array.__new__(GodotPoolVector2Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolVector2Array()
        elif is_godot_wrapper_instance(other, GodotPoolVector2Array):
            self._cpp_object = cpp.PoolVector2Array((<GodotPoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolVector2Array((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolVector3Array(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolVector3Array from_cpp(cpp.PoolVector3Array _cpp_object):
        cdef GodotPoolVector3Array self = GodotPoolVector3Array.__new__(GodotPoolVector3Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolVector3Array()
        elif is_godot_wrapper_instance(other, GodotPoolVector3Array):
            self._cpp_object = cpp.PoolVector3Array((<GodotPoolVector3Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolVector3Array((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolColorArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolColorArray from_cpp(cpp.PoolColorArray _cpp_object):
        cdef GodotPoolColorArray self = GodotPoolColorArray.__new__(GodotPoolColorArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolColorArray()
        elif is_godot_wrapper_instance(other, GodotPoolColorArray):
            self._cpp_object = cpp.PoolColorArray((<GodotPoolColorArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = cpp.PoolColorArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotQuat(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotQuat from_cpp(cpp.Quat _cpp_object):
        cdef GodotQuat self = GodotQuat.__new__(GodotQuat)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Quat()
        self._initialized = True


cdef class GodotRect2(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotRect2 from_cpp(cpp.Rect2 _cpp_object):
        cdef GodotRect2 self = GodotRect2.__new__(GodotRect2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, x=0, y=0, width=0, height=0):
        self._cpp_object = cpp.Rect2(x, y, width, height)
        self._initialized = True


cdef class GodotRID(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotRID from_cpp(cpp.RID _cpp_object):
        cdef GodotRID self = GodotRID.__new__(GodotRID)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, _Wrapped obj=None):
        cdef godot_object *p
        if obj is None:
            self._cpp_object = cpp.RID()
        else:
            p = obj._owner
            self._cpp_object = cpp.RID(<cpp.__Object *>p)
        self._initialized = True


cdef class GodotCharString(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotCharString from_cpp(cpp.CharString _cpp_object):
        cdef GodotCharString self = GodotCharString.__new__(GodotCharString)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def length(self):
        self._internal_check()
        return self._cpp_object.length()

    def get_data(self):
        self._internal_check()
        return <bytes>self._cpp_object.get_data()


cdef class GodotString(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotString from_cpp(cpp.String _cpp_object):
        cdef GodotString self = GodotString.__new__(GodotString)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object content=None):
        if not content:
            # Initialize an empty String for all falsy values
            self._cpp_object = cpp.String()
        elif isinstance(content, basestring):
            self._cpp_object = cpp.String(content)
        else:
            self._cpp_object = cpp.String(str(content))

    # @staticmethod
    # def num(double num, int decimals=-1):
    #     return GodotString.from_cpp(String.num(num, decimals))

    # @staticmethod
    # def num_scientific(double num):
    #     return GodotString.from_cpp(String.num_scientific(num))

    # @staticmethod
    # def num_real(double num):
    #     return GodotString.from_cpp(String.num_real(num))

    # @staticmethod
    # def num_int64(int64_t num, int base=10, bint capitalize_hex=False):
    #     return GodotString.from_cpp(String.num_int64(num, base, capitalize_hex))

    # TODO: chr, md5, hex_encode_buffer

    def __repr__(self):
        if self._initialized:
            return 'String(%r)' % self._cpp_object.py_str()
        return super().__repr__(self).replace('String', 'Uninitialized String')

    def __str__(self):
        if self._initialized:
            return self._cpp_object.py_str()
        return '<nil>'

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return <Py_UNICODE>self._cpp_object[item]

    # TODO


cdef class GodotTransform(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotTransform from_cpp(cpp.Transform _cpp_object):
        cdef GodotTransform self = GodotTransform.__new__(GodotTransform)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Transform()
        self._initialized = True


cdef class GodotTransform2D(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotTransform2D from_cpp(cpp.Transform2D _cpp_object):
        cdef GodotTransform2D self = GodotTransform2D.__new__(GodotTransform2D)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Transform2D()
        self._initialized = True


cdef class GodotVector2(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotVector2 from_cpp(cpp.Vector2 _cpp_object):
        cdef GodotVector2 self = GodotVector2.__new__(GodotVector2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0):
        self._cpp_object = cpp.Vector2(x, y)
        self._initialized = True


cdef class GodotVector3(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotVector3 from_cpp(cpp.Vector3 _cpp_object):
        cdef GodotVector3 self = GodotVector3.__new__(GodotVector3)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0, float z=0):
        self._cpp_object = cpp.Vector3(x, y, z)
        self._initialized = True


cdef public:
    # ctypedef GodotAABB _python_aabb_wrapper
    # ctypedef GodotArray _python_array_wrapper
    # ctypedef GodotBasis _python_basis_wrapper
    # ctypedef GodotColor _python_color_wrapper
    # ctypedef GodotDictionary _python_dictionary_wrapper
    # ctypedef GodotNodePath _python_nodepath_wrapper
    # ctypedef GodotPlane _python_plane_wrapper
    # ctypedef GodotPoolByteArray _python_poolbytearray_wrapper
    # ctypedef GodotPoolIntArray _python_poolintarray_wrapper
    # ctypedef GodotPoolRealArray _python_poolrealarray_wrapper
    # ctypedef GodotPoolStringArray _python_poolstringarray_wrapper
    # ctypedef GodotPoolVector2Array _python_poolvector2array_wrapper
    # ctypedef GodotPoolVector3Array _python_poolvector3array_wrapper
    # ctypedef GodotPoolColorArray _python_poolcolorarray_wrapper
    # ctypedef GodotQuat _python_quat_wrapper
    # ctypedef GodotRect2 _python_rect2_wrapper
    # ctypedef GodotRID _python_rid_wrapper
    # ctypedef GodotCharString _python_charstring_wrapper
    # ctypedef GodotString _python_string_wrapper
    # ctypedef GodotTransform _python_transform_wrapper
    # ctypedef GodotTransform2D _python_transform2d_wrapper
    # ctypedef GodotVector2 _python_vector2_wrapper
    # ctypedef GodotVector3 _python_vector3_wrapper

    cdef type PyGodotWrapperType_GodotAABB = GodotAABB
    cdef type PyGodotWrapperType_GodotArray = GodotArray
    cdef type PyGodotWrapperType_GodotBasis = GodotBasis
    cdef type PyGodotWrapperType_GodotColor = GodotColor
    cdef type PyGodotWrapperType_GodotDictionary = GodotDictionary
    cdef type PyGodotWrapperType_GodotNodePath = GodotNodePath
    cdef type PyGodotWrapperType_GodotPlane = GodotPlane
    cdef type PyGodotWrapperType_GodotPoolByteArray = GodotPoolByteArray
    cdef type PyGodotWrapperType_GodotPoolIntArray = GodotPoolIntArray
    cdef type PyGodotWrapperType_GodotPoolRealArray = GodotPoolRealArray
    cdef type PyGodotWrapperType_GodotPoolStringArray = GodotPoolStringArray
    cdef type PyGodotWrapperType_GodotPoolVector2Array = GodotPoolVector2Array
    cdef type PyGodotWrapperType_GodotPoolVector3Array = GodotPoolVector3Array
    cdef type PyGodotWrapperType_GodotPoolColorArray = GodotPoolColorArray
    cdef type PyGodotWrapperType_GodotQuat = GodotQuat
    cdef type PyGodotWrapperType_GodotRect2 = GodotRect2
    cdef type PyGodotWrapperType_GodotRID = GodotRID
    cdef type PyGodotWrapperType_GodotCharString = GodotCharString
    cdef type PyGodotWrapperType_GodotString = GodotString
    cdef type PyGodotWrapperType_GodotTransform = GodotTransform
    cdef type PyGodotWrapperType_GodotTransform2D = GodotTransform2D
    cdef type PyGodotWrapperType_GodotVector2 = GodotVector2
    cdef type PyGodotWrapperType_GodotVector3 = GodotVector3
    cdef type PyGodotType__Wrapped = _Wrapped

    object _aabb_to_python_wrapper(cpp.AABB _obj):
        return <object>GodotAABB.from_cpp(_obj)
    object _godot_array_to_python_wrapper(cpp.Array _obj):
        return <object>GodotArray.from_cpp(_obj)
    object _godot_basis_to_python_wrapper(cpp.Basis _obj):
        return <object>GodotBasis.from_cpp(_obj)
    object _color_to_python_wrapper(cpp.Color _obj):
        return <object>GodotColor.from_cpp(_obj)
    object _godot_dictionary_to_python_wrapper(cpp.Dictionary  _obj):
        return <object>GodotDictionary.from_cpp(_obj)
    object _nodepath_to_python_wrapper(cpp.NodePath _obj):
        return <object>GodotNodePath.from_cpp(_obj)
    object _plane_to_python_wrapper(cpp.Plane _obj):
        return <object>GodotPlane.from_cpp(_obj)
    object _poolbytearray_to_python_wrapper(cpp.PoolByteArray _obj):
        return <object>GodotPoolByteArray.from_cpp(_obj)
    object _poolintarray_to_python_wrapper(cpp.PoolIntArray _obj):
        return <object>GodotPoolIntArray.from_cpp(_obj)
    object _poolrealarray_to_python_wrapper(cpp.PoolRealArray _obj):
        return <object>GodotPoolRealArray.from_cpp(_obj)
    object _poolstringarray_to_python_wrapper(cpp.PoolStringArray _obj):
        return <object>GodotPoolStringArray.from_cpp(_obj)
    object _poolvector2array_to_python_wrapper(cpp.PoolVector2Array _obj):
        return <object>GodotPoolVector2Array.from_cpp(_obj)
    object _poolvector3array_to_python_wrapper(cpp.PoolVector3Array _obj):
        return <object>GodotPoolVector3Array.from_cpp(_obj)
    object _poolcolorarray_to_python_wrapper(cpp.PoolColorArray _obj):
        return <object>GodotPoolColorArray.from_cpp(_obj)
    object _quat_to_python_wrapper(cpp.Quat _obj):
        return <object>GodotQuat.from_cpp(_obj)
    object _rect2_to_python_wrapper(cpp.Rect2 _obj):
        return GodotRect2.from_cpp(_obj)
    object _rid_to_python_wrapper(cpp.RID _obj):
        return <object>GodotRID.from_cpp(_obj)
    object _charstring_to_python_wrapper(cpp.CharString _obj):
        return <object>GodotCharString.from_cpp(_obj)
    object _godot_string_to_python_wrapper(cpp.String _obj):
        return <object>GodotString.from_cpp(_obj)
    object _transform_to_python_wrapper(cpp.Transform _obj):
        return <object>GodotTransform.from_cpp(_obj)
    object _transform2d_to_python_wrapper(cpp.Transform2D _obj):
        return <object>GodotTransform2D.from_cpp(_obj)
    object _vector2_to_python_wrapper(cpp.Vector2 _obj):
        return <object>GodotVector2.from_cpp(_obj)
    object _vector3_to_python_wrapper(cpp.Vector3 _obj):
        return <object>GodotVector3.from_cpp(_obj)


    # Caller is responsible for type-checking in all
    # "*_binding_to_*" and "_python_wrapper_to_*" functions

    godot_object *_cython_binding_to_godot_object(object wrapped):
        return (<_Wrapped>wrapped)._owner

    godot_object *_python_binding_to_godot_object(object wrapped):
        return (<_PyWrapped>wrapped)._owner

    godot_array *_python_wrapper_to_godot_array(object wrapper):
        return <godot_array *>&(<GodotArray>wrapper)._cpp_object

    godot_vector2 *_python_wrapper_to_vector2(object wrapper):
        return <godot_vector2 *>&(<GodotVector2>wrapper)._cpp_object

