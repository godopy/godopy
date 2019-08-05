from godot_headers.gdnative_api cimport godot_object

from .cpp.core_types cimport *

from .globals cimport (
    gdapi,
    nativescript_1_1_api as ns11api,
    _nativescript_handle as handle,
    _cython_language_index as cython_idx,
    _python_language_index as python_idx
)


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
    cdef GodotAABB from_cpp(AABB _cpp_object):
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
            self._cpp_object = AABB((<GodotVector3>pos)._cpp_object, (<GodotVector3>size)._cpp_object)
        else:
            self._cpp_object = AABB()
            if pos is not None:
                self._cpp_object.set_position((<GodotVector3>pos)._cpp_object)
            if size is not None:
                self._cpp_object.set_size((<GodotVector3>size)._cpp_object)

        self._initialized = True



cdef class GodotArrayBase(GodotCoreTypeWrapper):
    pass


cdef class GodotArray(GodotArrayBase):
    @staticmethod
    cdef GodotArray from_cpp(Array _cpp_object):
        cdef GodotArray self = GodotArray.__new__(GodotArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        if other is None:
            self._cpp_object = Array()
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = Array((<GodotArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolByteArray):
            self._cpp_object = Array((<GodotPoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolIntArray):
            self._cpp_object = Array((<GodotPoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolRealArray):
            self._cpp_object = Array((<GodotPoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolStringArray):
            self._cpp_object = Array((<GodotPoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolVector2Array):
            self._cpp_object = Array((<GodotPoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolVector3Array):
            self._cpp_object = Array((<GodotPoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotPoolColorArray):
            self._cpp_object = Array((<GodotPoolByteArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


    @staticmethod
    def make(*values):
        array = GodotArray()

        for value in values:
            array._cpp_object.append(<const Variant &>value)

        return array

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return <object>self._cpp_object[<const int>item]

    # def __setitem__(self, int item, object value):
    #     self._internal_check()
    #     self._cpp_object[<const int>item] = <const Variant &>value


cdef class GodotBasis(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotBasis from_cpp(Basis _cpp_object):
        cdef GodotBasis self = GodotBasis.__new__(GodotBasis)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        self._cpp_object = Basis()
        self._initialized = True


cdef class GodotColor(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotColor from_cpp(Color _cpp_object):
        cdef GodotColor self = GodotColor.__new__(GodotColor)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float r=0, float g=0, float b=0, float a=1):
        self._cpp_object = Color(r, g, b, a)
        self._initialized = True

    @staticmethod
    def hex(self, uint32_t value):
        return GodotColor.from_cpp(Color.hex(value))

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return self._cpp_object[item]

    # def __setitem__(self, int item, float value):
    #     self._internal_check()
    #     self._cpp_object[item] = value


cdef class GodotDictionary(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotDictionary from_cpp(Dictionary _cpp_object):
        cdef GodotDictionary self = GodotDictionary.__new__(GodotDictionary)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = Dictionary()
        self._initialized = True


cdef class GodotNodePath(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotNodePath from_cpp(NodePath _cpp_object):
        cdef GodotNodePath self = GodotNodePath.__new__(GodotNodePath)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = NodePath()
        self._initialized = True


cdef class GodotPlane(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotPlane from_cpp(Plane _cpp_object):
        cdef GodotPlane self = GodotPlane.__new__(GodotPlane)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = Plane()
        self._initialized = True


cdef class GodotPoolArrayBase(GodotArrayBase):
    pass


cdef class GodotPoolByteArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolByteArray from_cpp(PoolByteArray _cpp_object):
        cdef GodotPoolByteArray self = GodotPoolByteArray.__new__(GodotPoolByteArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolByteArray()
        elif is_godot_wrapper_instance(other, GodotPoolByteArray):
            self._cpp_object = PoolByteArray((<GodotPoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolByteArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolIntArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolIntArray from_cpp(PoolIntArray _cpp_object):
        cdef GodotPoolIntArray self = GodotPoolIntArray.__new__(GodotPoolIntArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolIntArray()
        elif is_godot_wrapper_instance(other, GodotPoolIntArray):
            self._cpp_object = PoolIntArray((<GodotPoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolIntArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolRealArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolRealArray from_cpp(PoolRealArray _cpp_object):
        cdef GodotPoolRealArray self = GodotPoolRealArray.__new__(GodotPoolRealArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolRealArray()
        elif is_godot_wrapper_instance(other, GodotPoolRealArray):
            self._cpp_object = PoolRealArray((<GodotPoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolRealArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)
        self._initialized = True


cdef class GodotPoolStringArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolStringArray from_cpp(PoolStringArray _cpp_object):
        cdef GodotPoolStringArray self = GodotPoolStringArray.__new__(GodotPoolStringArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolStringArray()
        elif is_godot_wrapper_instance(other, GodotPoolStringArray):
            self._cpp_object = PoolStringArray((<GodotPoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolStringArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolVector2Array(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolVector2Array from_cpp(PoolVector2Array _cpp_object):
        cdef GodotPoolVector2Array self = GodotPoolVector2Array.__new__(GodotPoolVector2Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolVector2Array()
        elif is_godot_wrapper_instance(other, GodotPoolVector2Array):
            self._cpp_object = PoolVector2Array((<GodotPoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolVector2Array((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolVector3Array(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolVector3Array from_cpp(PoolVector3Array _cpp_object):
        cdef GodotPoolVector3Array self = GodotPoolVector3Array.__new__(GodotPoolVector3Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolVector3Array()
        elif is_godot_wrapper_instance(other, GodotPoolVector3Array):
            self._cpp_object = PoolVector3Array((<GodotPoolVector3Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolVector3Array((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotPoolColorArray(GodotPoolArrayBase):
    @staticmethod
    cdef GodotPoolColorArray from_cpp(PoolColorArray _cpp_object):
        cdef GodotPoolColorArray self = GodotPoolColorArray.__new__(GodotPoolColorArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = PoolColorArray()
        elif is_godot_wrapper_instance(other, GodotPoolColorArray):
            self._cpp_object = PoolColorArray((<GodotPoolColorArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, GodotArray):
            self._cpp_object = PoolColorArray((<GodotArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class GodotQuat(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotQuat from_cpp(Quat _cpp_object):
        cdef GodotQuat self = GodotQuat.__new__(GodotQuat)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = Quat()
        self._initialized = True


cdef class GodotRect2(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotRect2 from_cpp(Rect2 _cpp_object):
        cdef GodotRect2 self = GodotRect2.__new__(GodotRect2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, x=0, y=0, width=0, height=0):
        self._cpp_object = Rect2(x, y, width, height)
        self._initialized = True


cdef class GodotRID(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotRID from_cpp(RID _cpp_object):
        cdef GodotRID self = GodotRID.__new__(GodotRID)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, _Wrapped obj=None):
        cdef godot_object *p
        if obj is None:
            self._cpp_object = RID()
        else:
            p = obj._owner
            self._cpp_object = RID(<__Object *>p)
        self._initialized = True


cdef class GodotCharString(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotCharString from_cpp(CharString _cpp_object):
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
    cdef GodotString from_cpp(String _cpp_object):
        cdef GodotString self = GodotString.__new__(GodotString)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object content=None):
        if not content:
            # Initialize an empty String for all falsy values
            self._cpp_object = String()
        elif isinstance(content, basestring):
            self._cpp_object = String(content)
        else:
            self._cpp_object = String(str(content))

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
            return 'GodotString(%r)' % self._cpp_object.py_str()
        return super().__repr__(self).replace('GodotString', 'Uninitialized GodotString')

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
    cdef GodotTransform from_cpp(Transform _cpp_object):
        cdef GodotTransform self = GodotTransform.__new__(GodotTransform)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = Transform()
        self._initialized = True


cdef class GodotTransform2D(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotTransform2D from_cpp(Transform2D _cpp_object):
        cdef GodotTransform2D self = GodotTransform2D.__new__(GodotTransform2D)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = Transform2D()
        self._initialized = True


cdef class GodotVector2(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotVector2 from_cpp(Vector2 _cpp_object):
        cdef GodotVector2 self = GodotVector2.__new__(GodotVector2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0):
        self._cpp_object = Vector2(x, y)
        self._initialized = True


cdef class GodotVector3(GodotCoreTypeWrapper):
    @staticmethod
    cdef GodotVector3 from_cpp(Vector3 _cpp_object):
        cdef GodotVector3 self = GodotVector3.__new__(GodotVector3)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0, float z=0):
        self._cpp_object = Vector3(x, y, z)
        self._initialized = True


cdef class _Wrapped:
    pass


cdef class _PyWrapped:
    pass


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

    cdef type PyGodotType_GodotAABB = GodotAABB
    cdef type PyGodotType_GodotArray = GodotArray
    cdef type PyGodotType_GodotBasis = GodotBasis
    cdef type PyGodotType_GodotColor = GodotColor
    cdef type PyGodotType_GodotDictionary = GodotDictionary
    cdef type PyGodotType_GodotNodePath = GodotNodePath
    cdef type PyGodotType_GodotPlane = GodotPlane
    cdef type PyGodotType_GodotPoolByteArray = GodotPoolByteArray
    cdef type PyGodotType_GodotPoolIntArray = GodotPoolIntArray
    cdef type PyGodotType_GodotPoolRealArray = GodotPoolRealArray
    cdef type PyGodotType_GodotPoolStringArray = GodotPoolStringArray
    cdef type PyGodotType_GodotPoolVector2Array = GodotPoolVector2Array
    cdef type PyGodotType_GodotPoolVector3Array = GodotPoolVector3Array
    cdef type PyGodotType_GodotPoolColorArray = GodotPoolColorArray
    cdef type PyGodotType_GodotQuat = GodotQuat
    cdef type PyGodotType_GodotRect2 = GodotRect2
    cdef type PyGodotType_GodotRID = GodotRID
    cdef type PyGodotType_GodotCharString = GodotCharString
    cdef type PyGodotType_GodotString = GodotString
    cdef type PyGodotType_GodotTransform = GodotTransform
    cdef type PyGodotType_GodotTransform2D = GodotTransform2D
    cdef type PyGodotType_GodotVector2 = GodotVector2
    cdef type PyGodotType_GodotVector3 = GodotVector3
    cdef type PyGodotType__Wrapped = _Wrapped

    object _aabb_to_python(AABB _obj):
        return <object>GodotAABB.from_cpp(_obj)
    object _godot_array_to_python(Array _obj):
        return <object>GodotArray.from_cpp(_obj)
    object _godot_basis_to_python(Basis _obj):
        return <object>GodotBasis.from_cpp(_obj)
    object _color_to_python(Color _obj):
        return <object>GodotColor.from_cpp(_obj)
    object _godot_dictionary_to_python(Dictionary  _obj):
        return <object>GodotDictionary.from_cpp(_obj)
    object _nodepath_to_python(NodePath _obj):
        return <object>GodotNodePath.from_cpp(_obj)
    object _plane_to_python(Plane _obj):
        return <object>GodotPlane.from_cpp(_obj)
    object _poolbytearray_to_python(PoolByteArray _obj):
        return <object>GodotPoolByteArray.from_cpp(_obj)
    object _poolintarray_to_python(PoolIntArray _obj):
        return <object>GodotPoolIntArray.from_cpp(_obj)
    object _poolrealarray_to_python(PoolRealArray _obj):
        return <object>GodotPoolRealArray.from_cpp(_obj)
    object _poolstringarray_to_python(PoolStringArray _obj):
        return <object>GodotPoolStringArray.from_cpp(_obj)
    object _poolvector2array_to_python(PoolVector2Array _obj):
        return <object>GodotPoolVector2Array.from_cpp(_obj)
    object _poolvector3array_to_python(PoolVector3Array _obj):
        return <object>GodotPoolVector3Array.from_cpp(_obj)
    object _poolcolorarray_to_python(PoolColorArray _obj):
        return <object>GodotPoolColorArray.from_cpp(_obj)
    object _quat_to_python(Quat _obj):
        return <object>GodotQuat.from_cpp(_obj)
    object _rect2_to_python(Rect2 _obj):
        return GodotRect2.from_cpp(_obj)
    object _rid_to_python(RID _obj):
        return <object>GodotRID.from_cpp(_obj)
    object _charstring_to_python(CharString _obj):
        return <object>GodotCharString.from_cpp(_obj)
    object _godot_string_to_python(String _obj):
        return <object>GodotString.from_cpp(_obj)
    object _transform_to_python(Transform _obj):
        return <object>GodotTransform.from_cpp(_obj)
    object _transform2d_to_python(Transform2D _obj):
        return <object>GodotTransform2D.from_cpp(_obj)
    object  _vector2_to_python(Vector2 _obj):
        return <object>GodotVector2.from_cpp(_obj)
    object _vector3_to_python(Vector3 _obj):
        return <object>GodotVector3.from_cpp(_obj)


    godot_object *_python_to_godot_object(object wrapped):
        return (<_Wrapped>wrapped)._owner

    godot_array *_python_to_godot_array(object wrapper):
        return <godot_array *>&(<GodotArray>wrapper)._cpp_object

    godot_vector2 *_python_to_vector2(object wrapper):
        return <godot_vector2 *>&(<GodotVector2>wrapper)._cpp_object


include "core/signal_arguments_impl.pxi"

cdef dict CythonTagDB = {}
cdef dict PythonTagDB = {}
cdef dict __instance_map = {}


cdef register_cython_type(type cls):
    cdef size_t type_tag = <size_t><void *>cls
    cdef bytes name = cls.__name__.encode('utf-8')

    print('set type tag', name, type_tag)

    ns11api.godot_nativescript_set_type_tag(handle, <const char *>name, <void *>type_tag)

    CythonTagDB[type_tag] = cls


cdef register_python_type(type cls):
    cdef size_t type_tag = <size_t><void *>cls
    cdef bytes name = cls.__name__.encode('utf-8')

    ns11api.godot_nativescript_set_type_tag(handle, <const char *>name, <void *>type_tag)

    PythonTagDB[type_tag] = cls


cdef register_global_cython_type(type cls, str api_name):
    cdef bytes _api_name = api_name.encode('utf-8')
    cdef size_t type_tag = <size_t><void *>cls

    ns11api.godot_nativescript_set_global_type_tag(cython_idx, <const char *>_api_name, <const void *>type_tag)

    CythonTagDB[type_tag] = cls


cdef register_global_python_type(type cls, str api_name):
    cdef bytes _api_name = api_name.encode('utf-8')
    cdef size_t type_tag = <size_t><void *>cls

    ns11api.godot_nativescript_set_global_type_tag(python_idx, <const char *>_api_name, <const void *>type_tag)

    PythonTagDB[type_tag] = cls

    # cls.__godot_api_name__ = api_name


cdef get_instance_from_owner(godot_object *instance):
    return __instance_map[<size_t>instance]
