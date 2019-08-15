from libc.stdint cimport uint64_t, uint32_t, uint8_t, int64_t, int32_t
from libc.stddef cimport wchar_t

from godot_headers.gdnative_api cimport godot_object, godot_array, godot_vector2

from ._wrapped cimport _Wrapped, _PyWrapped
from . cimport cpp_types as cpp


def is_godot_wrapper_instance(object obj, object instances):
    return isinstance(obj, instances) and (<CoreTypeWrapper?>obj)._initialized

cdef class CoreTypeWrapper:
    def __cinit__(self):
        self._initialized = False


cdef class AABB(CoreTypeWrapper):
    @staticmethod
    cdef AABB from_cpp(cpp.AABB _cpp_object):
        cdef AABB self = AABB.__new__(AABB)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object pos=None, object size=None):
        if pos is not None and not is_godot_wrapper_instance(pos, Vector3):
           raise self._init_value_error(pos)
        if size is not None and not is_godot_wrapper_instance(pos, Vector3):
            raise self._init_value_error(size)

        if pos is not None and size is not None:
            self._cpp_object = cpp.AABB((<Vector3>pos)._cpp_object, (<Vector3>size)._cpp_object)
        else:
            self._cpp_object = cpp.AABB()
            if pos is not None:
                self._cpp_object.set_position((<Vector3>pos)._cpp_object)
            if size is not None:
                self._cpp_object.set_size((<Vector3>size)._cpp_object)

        self._initialized = True


cdef class ArrayBase(CoreTypeWrapper):
    pass


cdef class Array(ArrayBase):
    @staticmethod
    cdef Array from_cpp(cpp.Array _cpp_object):
        cdef Array self = Array.__new__(Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        if other is None:
            self._cpp_object = cpp.Array()
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.Array((<Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolByteArray):
            self._cpp_object = cpp.Array((<PoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolIntArray):
            self._cpp_object = cpp.Array((<PoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolRealArray):
            self._cpp_object = cpp.Array((<PoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolStringArray):
            self._cpp_object = cpp.Array((<PoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolVector2Array):
            self._cpp_object = cpp.Array((<PoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolVector3Array):
            self._cpp_object = cpp.Array((<PoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolColorArray):
            self._cpp_object = cpp.Array((<PoolByteArray>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


    @staticmethod
    def make(*values):
        array = Array()

        for value in values:
            array._cpp_object.append(<const cpp.Variant &>value)

        return array

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return <object>self._cpp_object[<const int>item]

    # def __setitem__(self, int item, object value):
    #     self._internal_check()
    #     self._cpp_object[<const int>item] = <const cpp.Variant &>value


cdef class Basis(CoreTypeWrapper):
    @staticmethod
    cdef Basis from_cpp(cpp.Basis _cpp_object):
        cdef Basis self = Basis.__new__(Basis)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        self._cpp_object = cpp.Basis()
        self._initialized = True


cdef class Color(CoreTypeWrapper):
    @staticmethod
    cdef Color from_cpp(cpp.Color _cpp_object):
        cdef Color self = Color.__new__(Color)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float r=0, float g=0, float b=0, float a=1):
        self._cpp_object = cpp.Color(r, g, b, a)
        self._initialized = True

    @staticmethod
    def hex(self, uint32_t value):
        return Color.from_cpp(cpp.Color.hex(value))

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return self._cpp_object[item]

    # def __setitem__(self, int item, float value):
    #     self._internal_check()
    #     self._cpp_object[item] = value


cdef class Dictionary(CoreTypeWrapper):
    @staticmethod
    cdef Dictionary from_cpp(cpp.Dictionary _cpp_object):
        cdef Dictionary self = Dictionary.__new__(Dictionary)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Dictionary()
        self._initialized = True


cdef class NodePath(CoreTypeWrapper):
    @staticmethod
    cdef NodePath from_cpp(cpp.NodePath _cpp_object):
        cdef NodePath self = NodePath.__new__(NodePath)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.NodePath()
        self._initialized = True


cdef class Plane(CoreTypeWrapper):
    @staticmethod
    cdef Plane from_cpp(cpp.Plane _cpp_object):
        cdef Plane self = Plane.__new__(Plane)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Plane()
        self._initialized = True


cdef class PoolArrayBase(ArrayBase):
    pass


cdef class PoolByteArray(PoolArrayBase):
    @staticmethod
    cdef PoolByteArray from_cpp(cpp.PoolByteArray _cpp_object):
        cdef PoolByteArray self = PoolByteArray.__new__(PoolByteArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolByteArray()
        elif is_godot_wrapper_instance(other, PoolByteArray):
            self._cpp_object = cpp.PoolByteArray((<PoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolByteArray((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class PoolIntArray(PoolArrayBase):
    @staticmethod
    cdef PoolIntArray from_cpp(cpp.PoolIntArray _cpp_object):
        cdef PoolIntArray self = PoolIntArray.__new__(PoolIntArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolIntArray()
        elif is_godot_wrapper_instance(other, PoolIntArray):
            self._cpp_object = cpp.PoolIntArray((<PoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolIntArray((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class PoolRealArray(PoolArrayBase):
    @staticmethod
    cdef PoolRealArray from_cpp(cpp.PoolRealArray _cpp_object):
        cdef PoolRealArray self = PoolRealArray.__new__(PoolRealArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolRealArray()
        elif is_godot_wrapper_instance(other, PoolRealArray):
            self._cpp_object = cpp.PoolRealArray((<PoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolRealArray((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)
        self._initialized = True


cdef class PoolStringArray(PoolArrayBase):
    @staticmethod
    cdef PoolStringArray from_cpp(cpp.PoolStringArray _cpp_object):
        cdef PoolStringArray self = PoolStringArray.__new__(PoolStringArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolStringArray()
        elif is_godot_wrapper_instance(other, PoolStringArray):
            self._cpp_object = cpp.PoolStringArray((<PoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolStringArray((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class PoolVector2Array(PoolArrayBase):
    @staticmethod
    cdef PoolVector2Array from_cpp(cpp.PoolVector2Array _cpp_object):
        cdef PoolVector2Array self = PoolVector2Array.__new__(PoolVector2Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolVector2Array()
        elif is_godot_wrapper_instance(other, PoolVector2Array):
            self._cpp_object = cpp.PoolVector2Array((<PoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolVector2Array((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class PoolVector3Array(PoolArrayBase):
    @staticmethod
    cdef PoolVector3Array from_cpp(cpp.PoolVector3Array _cpp_object):
        cdef PoolVector3Array self = PoolVector3Array.__new__(PoolVector3Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolVector3Array()
        elif is_godot_wrapper_instance(other, PoolVector3Array):
            self._cpp_object = cpp.PoolVector3Array((<PoolVector3Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolVector3Array((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class PoolColorArray(PoolArrayBase):
    @staticmethod
    cdef PoolColorArray from_cpp(cpp.PoolColorArray _cpp_object):
        cdef PoolColorArray self = PoolColorArray.__new__(PoolColorArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolColorArray()
        elif is_godot_wrapper_instance(other, PoolColorArray):
            self._cpp_object = cpp.PoolColorArray((<PoolColorArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolColorArray((<Array>other)._cpp_object)
        else:
            raise self._init_value_error(other)

        self._initialized = True


cdef class Quat(CoreTypeWrapper):
    @staticmethod
    cdef Quat from_cpp(cpp.Quat _cpp_object):
        cdef Quat self = Quat.__new__(Quat)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Quat()
        self._initialized = True


cdef class Rect2(CoreTypeWrapper):
    @staticmethod
    cdef Rect2 from_cpp(cpp.Rect2 _cpp_object):
        cdef Rect2 self = Rect2.__new__(Rect2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, x=0, y=0, width=0, height=0):
        self._cpp_object = cpp.Rect2(x, y, width, height)
        self._initialized = True


cdef class RID(CoreTypeWrapper):
    @staticmethod
    cdef RID from_cpp(cpp.RID _cpp_object):
        cdef RID self = RID.__new__(RID)
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


cdef class CharString(CoreTypeWrapper):
    @staticmethod
    cdef CharString from_cpp(cpp.CharString _cpp_object):
        cdef CharString self = CharString.__new__(CharString)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def length(self):
        self._internal_check()
        return self._cpp_object.length()

    def get_data(self):
        self._internal_check()
        return <bytes>self._cpp_object.get_data()


cdef class String(CoreTypeWrapper):
    @staticmethod
    cdef String from_cpp(cpp.String _cpp_object):
        cdef String self = String.__new__(String)
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
    #     return String.from_cpp(String.num(num, decimals))

    # @staticmethod
    # def num_scientific(double num):
    #     return String.from_cpp(String.num_scientific(num))

    # @staticmethod
    # def num_real(double num):
    #     return String.from_cpp(String.num_real(num))

    # @staticmethod
    # def num_int64(int64_t num, int base=10, bint capitalize_hex=False):
    #     return String.from_cpp(String.num_int64(num, base, capitalize_hex))

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


cdef class Transform(CoreTypeWrapper):
    @staticmethod
    cdef Transform from_cpp(cpp.Transform _cpp_object):
        cdef Transform self = Transform.__new__(Transform)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Transform()
        self._initialized = True


cdef class Transform2D(CoreTypeWrapper):
    @staticmethod
    cdef Transform2D from_cpp(cpp.Transform2D _cpp_object):
        cdef Transform2D self = Transform2D.__new__(Transform2D)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Transform2D()
        self._initialized = True


cdef class Vector2(CoreTypeWrapper):
    @staticmethod
    cdef Vector2 from_cpp(cpp.Vector2 _cpp_object):
        cdef Vector2 self = Vector2.__new__(Vector2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0):
        self._cpp_object = cpp.Vector2(x, y)
        self._initialized = True


cdef class Vector3(CoreTypeWrapper):
    @staticmethod
    cdef Vector3 from_cpp(cpp.Vector3 _cpp_object):
        cdef Vector3 self = Vector3.__new__(Vector3)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0, float z=0):
        self._cpp_object = cpp.Vector3(x, y, z)
        self._initialized = True


cdef public:
    # ctypedef AABB _python_aabb_wrapper
    # ctypedef Array _python_array_wrapper
    # ctypedef Basis _python_basis_wrapper
    # ctypedef Color _python_color_wrapper
    # ctypedef Dictionary _python_dictionary_wrapper
    # ctypedef NodePath _python_nodepath_wrapper
    # ctypedef Plane _python_plane_wrapper
    # ctypedef PoolByteArray _python_poolbytearray_wrapper
    # ctypedef PoolIntArray _python_poolintarray_wrapper
    # ctypedef PoolRealArray _python_poolrealarray_wrapper
    # ctypedef PoolStringArray _python_poolstringarray_wrapper
    # ctypedef PoolVector2Array _python_poolvector2array_wrapper
    # ctypedef PoolVector3Array _python_poolvector3array_wrapper
    # ctypedef PoolColorArray _python_poolcolorarray_wrapper
    # ctypedef Quat _python_quat_wrapper
    # ctypedef Rect2 _python_rect2_wrapper
    # ctypedef RID _python_rid_wrapper
    # ctypedef CharString _python_charstring_wrapper
    # ctypedef String _python_string_wrapper
    # ctypedef Transform _python_transform_wrapper
    # ctypedef Transform2D _python_transform2d_wrapper
    # ctypedef Vector2 _python_vector2_wrapper
    # ctypedef Vector3 _python_vector3_wrapper

    cdef type PyGodotWrapperType_AABB = AABB
    cdef type PyGodotWrapperType_Array = Array
    cdef type PyGodotWrapperType_Basis = Basis
    cdef type PyGodotWrapperType_Color = Color
    cdef type PyGodotWrapperType_Dictionary = Dictionary
    cdef type PyGodotWrapperType_NodePath = NodePath
    cdef type PyGodotWrapperType_Plane = Plane
    cdef type PyGodotWrapperType_PoolByteArray = PoolByteArray
    cdef type PyGodotWrapperType_PoolIntArray = PoolIntArray
    cdef type PyGodotWrapperType_PoolRealArray = PoolRealArray
    cdef type PyGodotWrapperType_PoolStringArray = PoolStringArray
    cdef type PyGodotWrapperType_PoolVector2Array = PoolVector2Array
    cdef type PyGodotWrapperType_PoolVector3Array = PoolVector3Array
    cdef type PyGodotWrapperType_PoolColorArray = PoolColorArray
    cdef type PyGodotWrapperType_Quat = Quat
    cdef type PyGodotWrapperType_Rect2 = Rect2
    cdef type PyGodotWrapperType_RID = RID
    cdef type PyGodotWrapperType_CharString = CharString
    cdef type PyGodotWrapperType_String = String
    cdef type PyGodotWrapperType_Transform = Transform
    cdef type PyGodotWrapperType_Transform2D = Transform2D
    cdef type PyGodotWrapperType_Vector2 = Vector2
    cdef type PyGodotWrapperType_Vector3 = Vector3
    cdef type PyGodotType__Wrapped = _Wrapped

    object _aabb_to_python_wrapper(cpp.AABB _obj):
        return <object>AABB.from_cpp(_obj)
    object _godot_array_to_python_wrapper(cpp.Array _obj):
        return <object>Array.from_cpp(_obj)
    object _godot_basis_to_python_wrapper(cpp.Basis _obj):
        return <object>Basis.from_cpp(_obj)
    object _color_to_python_wrapper(cpp.Color _obj):
        return <object>Color.from_cpp(_obj)
    object _godot_dictionary_to_python_wrapper(cpp.Dictionary  _obj):
        return <object>Dictionary.from_cpp(_obj)
    object _nodepath_to_python_wrapper(cpp.NodePath _obj):
        return <object>NodePath.from_cpp(_obj)
    object _plane_to_python_wrapper(cpp.Plane _obj):
        return <object>Plane.from_cpp(_obj)
    object _poolbytearray_to_python_wrapper(cpp.PoolByteArray _obj):
        return <object>PoolByteArray.from_cpp(_obj)
    object _poolintarray_to_python_wrapper(cpp.PoolIntArray _obj):
        return <object>PoolIntArray.from_cpp(_obj)
    object _poolrealarray_to_python_wrapper(cpp.PoolRealArray _obj):
        return <object>PoolRealArray.from_cpp(_obj)
    object _poolstringarray_to_python_wrapper(cpp.PoolStringArray _obj):
        return <object>PoolStringArray.from_cpp(_obj)
    object _poolvector2array_to_python_wrapper(cpp.PoolVector2Array _obj):
        return <object>PoolVector2Array.from_cpp(_obj)
    object _poolvector3array_to_python_wrapper(cpp.PoolVector3Array _obj):
        return <object>PoolVector3Array.from_cpp(_obj)
    object _poolcolorarray_to_python_wrapper(cpp.PoolColorArray _obj):
        return <object>PoolColorArray.from_cpp(_obj)
    object _quat_to_python_wrapper(cpp.Quat _obj):
        return <object>Quat.from_cpp(_obj)
    object _rect2_to_python_wrapper(cpp.Rect2 _obj):
        return Rect2.from_cpp(_obj)
    object _rid_to_python_wrapper(cpp.RID _obj):
        return <object>RID.from_cpp(_obj)
    object _charstring_to_python_wrapper(cpp.CharString _obj):
        return <object>CharString.from_cpp(_obj)
    object _godot_string_to_python_wrapper(cpp.String _obj):
        return <object>String.from_cpp(_obj)
    object _transform_to_python_wrapper(cpp.Transform _obj):
        return <object>Transform.from_cpp(_obj)
    object _transform2d_to_python_wrapper(cpp.Transform2D _obj):
        return <object>Transform2D.from_cpp(_obj)
    object _vector2_to_python_wrapper(cpp.Vector2 _obj):
        return <object>Vector2.from_cpp(_obj)
    object _vector3_to_python_wrapper(cpp.Vector3 _obj):
        return <object>Vector3.from_cpp(_obj)


    # Caller is responsible for type-checking in all
    # "*_binding_to_*" and "_python_wrapper_to_*" functions

    godot_object *_cython_binding_to_godot_object(object wrapped):
        return (<_Wrapped>wrapped)._owner

    godot_object *_python_binding_to_godot_object(object wrapped):
        return (<_PyWrapped>wrapped)._owner

    godot_array *_python_wrapper_to_godot_array(object wrapper):
        return <godot_array *>&(<Array>wrapper)._cpp_object

    godot_vector2 *_python_wrapper_to_vector2(object wrapper):
        return <godot_vector2 *>&(<Vector2>wrapper)._cpp_object

