from gdextension_interface cimport *
from binding cimport *
from libc.stdint cimport *
from libc.stddef cimport wchar_t
cimport cpython
cimport godot_cpp as cpp
from gdextension cimport ArgType, Object, Callable

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy


import numpy as np


ctypedef object (*variant_to_pyobject_func_t)(const cpp.Variant &)
ctypedef void (*variant_from_pyobject_func_t)(object, cpp.Variant *) noexcept


cdef inline bint issubscriptable(object obj):
    return numpy.PyArray_Check(obj) or cpython.PySequence_Check(obj)


cdef inline bint isstring_dtype(object dtype):
    return np.issubdtype(dtype, np.character) or np.issubdtype(dtype, np.dtypes.StringDType) \
           or np.issubdtype(dtype, np.dtypes.BytesDType)


cdef public object variant_nil_to_pyobject(const cpp.Variant &v)
cdef public void variant_nil_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object bool_to_pyobject(const uint8_t p_bool)
cdef public object variant_bool_to_pyobject(const cpp.Variant &v)
cdef public void bool_from_pyobject(object p_obj, uint8_t *r_ret) noexcept
cdef public void variant_bool_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object int_to_pyobject(const int64_t p_int)
cdef public object variant_int_to_pyobject(const cpp.Variant &v)
cdef public void int_from_pyobject(object p_obj, int64_t *r_ret) noexcept
cdef public void variant_int_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object float_to_pyobject(const double p_float)
cdef public object variant_float_to_pyobject(const cpp.Variant &v)
cdef public void float_from_pyobject(object p_obj, double *r_ret) noexcept
cdef public void variant_float_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef class String(str): pass
cdef public object string_to_pyobject(const cpp.String &p_string)
cdef public object variant_string_to_pyobject(const cpp.Variant &v)
cdef public void string_from_pyobject(object p_obj, cpp.String *r_ret) noexcept
cdef public void variant_string_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object vector2_to_pyobject(const cpp.Vector2 &vec)
cdef public object vector2i_to_pyobject(const cpp.Vector2i &vec)
cdef public object variant_vector2_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector2i_to_pyobject(const cpp.Variant &v)
cdef public void vector2_from_pyobject(object p_obj, cpp.Vector2 *r_ret) noexcept
cdef public void vector2i_from_pyobject(object p_obj, cpp.Vector2i *r_ret) noexcept
cdef public void variant_vector2_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector2i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object rect2_to_pyobject(const cpp.Rect2 &rect)
cdef public object rect2i_to_pyobject(const cpp.Rect2i &rect)
cdef public object variant_rect2_to_pyobject(const cpp.Variant &v)
cdef public object variant_rect2i_to_pyobject(const cpp.Variant &v)
cdef public void rect2_from_pyobject(object p_obj, cpp.Rect2 *r_ret) noexcept
cdef public void rect2i_from_pyobject(object p_obj, cpp.Rect2i *r_ret) noexcept
cdef public void variant_rect2_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_rect2i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object vector3_to_pyobject(const cpp.Vector3 &vec)
cdef public object vector3i_to_pyobject(const cpp.Vector3i &vec)
cdef public object variant_vector3_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector3i_to_pyobject(const cpp.Variant &v)
cdef public void vector3_from_pyobject(object p_obj, cpp.Vector3 *r_ret) noexcept
cdef public void vector3i_from_pyobject(object p_obj, cpp.Vector3i *r_ret) noexcept
cdef public void variant_vector3_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector3i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object transform2d_to_pyobject(const cpp.Transform2D &t)
cdef public object variant_transform2d_to_pyobject(const cpp.Variant &v)
cdef public void transform2d_from_pyobject(object p_obj, cpp.Transform2D *r_ret) noexcept
cdef public void variant_transform2d_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object vector4_to_pyobject(const cpp.Vector4 &vec)
cdef public object vector4i_to_pyobject(const cpp.Vector4i &vec)
cdef public object variant_vector4_to_pyobject(const cpp.Variant &v)
cdef public object variant_vector4i_to_pyobject(const cpp.Variant &v)
cdef public void vector4_from_pyobject(object p_obj, cpp.Vector4 *r_ret) noexcept
cdef public void vector4i_from_pyobject(object p_obj, cpp.Vector4i *r_ret) noexcept
cdef public void variant_vector4_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_vector4i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object plane_to_pyobject(const cpp.Plane &plane)
cdef public object variant_plane_to_pyobject(const cpp.Variant &v)
cdef public void plane_from_pyobject(object p_obj, cpp.Plane *r_ret) noexcept
cdef public void variant_plane_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object quaternion_to_pyobject(const cpp.Quaternion &q)
cdef public object variant_quaternion_to_pyobject(const cpp.Variant &v)
cdef public void quaternion_from_pyobject(object p_obj, cpp.Quaternion *r_ret) noexcept
cdef public void variant_quaternion_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object aabb_to_pyobject(const cpp._AABB &p_aabb)
cdef public object variant_aabb_to_pyobject(const cpp.Variant &v)
cdef public void aabb_from_pyobject(object p_obj, cpp._AABB *r_ret) noexcept
cdef public void variant_aabb_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object basis_to_pyobject(const cpp.Basis &b)
cdef public object variant_basis_to_pyobject(const cpp.Variant &v)
cdef public void basis_from_pyobject(object p_obj, cpp.Basis *r_ret) noexcept
cdef public void variant_basis_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object transform3d_to_pyobject(const cpp.Transform3D &t)
cdef public object variant_transform3d_to_pyobject(const cpp.Variant &v)
cdef public void transform3d_from_pyobject(object p_obj, cpp.Transform3D *r_ret) noexcept
cdef public void variant_transform3d_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object projection_to_pyobject(const cpp.Projection &p)
cdef public object variant_projection_to_pyobject(const cpp.Variant &v)
cdef public void projection_from_pyobject(object p_obj, cpp.Projection *r_ret) noexcept
cdef public void variant_projection_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object color_to_pyobject(const cpp.Color &p_color)
cdef public object variant_color_to_pyobject(const cpp.Variant &v)
cdef public void color_from_pyobject(object p_obj, cpp.Color *r_ret) noexcept
cdef public void variant_color_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept


cdef class StringName(str):
    # TODO: Try to use GDExtension API directly without godot-cpp objects
    cdef cpp.StringName _base
    cdef void *ptr(self)

cdef public object string_name_to_pyobject(const cpp.StringName &p_val)
cdef public object variant_string_name_to_pyobject(const cpp.Variant &v)
cdef public void string_name_from_pyobject(object p_obj, cpp.StringName *r_ret) noexcept
cdef public void variant_string_name_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object node_path_to_pyobject(const cpp.NodePath &p_val)
cdef public object variant_node_path_to_pyobject(const cpp.Variant &v)
cdef public void node_path_from_pyobject(object p_obj, cpp.NodePath *r_ret) noexcept
cdef public void variant_node_path_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept


cdef class RID:
    # TODO: Try to use GDExtension API directly without godot-cpp objects
    cdef cpp._RID _base

    @staticmethod
    cdef RID from_cpp_rid(const cpp._RID &p_val)

cdef public object rid_to_pyobject(const cpp._RID &p_val)
cdef public object variant_rid_to_pyobject(const cpp.Variant &v)
cdef public void rid_from_pyobject(object p_obj, cpp._RID *r_ret) noexcept
cdef public void variant_rid_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept


cdef class Signal:
    # TODO: Try to use GDExtension API directly without godot-cpp objects
    cdef cpp.GodotCppSignal _godot_signal

    @staticmethod
    cdef Signal from_cpp(const cpp.GodotCppSignal &)

cdef public object signal_to_pyobject(const cpp.GodotCppSignal &s)
cdef public object variant_signal_to_pyobject(const cpp.Variant &v)
cdef public void signal_from_pyobject(object p_obj, cpp.GodotCppSignal *r_ret) noexcept
cdef public void variant_signal_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object dictionary_to_pyobject(const cpp.Dictionary &p_val)
cdef public object variant_dictionary_to_pyobject(const cpp.Variant &v)
cdef public void dictionary_from_pyobject(object p_obj, cpp.Dictionary *r_ret) noexcept
cdef public void variant_dictionary_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object array_to_pyobject(const cpp.Array &p_arr)
cdef public object variant_array_to_pyobject(const cpp.Variant &v)
cdef public void array_from_pyobject(object p_obj, cpp.Array *r_ret) noexcept
cdef public void variant_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object packed_byte_array_to_pyobject(const cpp.PackedByteArray &p_arr)
cdef public object packed_int32_array_to_pyobject(const cpp.PackedInt32Array &p_arr)
cdef public object packed_int64_array_to_pyobject(const cpp.PackedInt64Array &p_arr)
cdef public object packed_float32_array_to_pyobject(const cpp.PackedFloat32Array &p_arr)
cdef public object packed_float64_array_to_pyobject(const cpp.PackedFloat64Array &p_arr)
cdef public object packed_string_array_to_pyobject(const cpp.PackedStringArray &p_arr)
cdef public object packed_vector2_array_to_pyobject(const cpp.PackedVector2Array &p_arr)
cdef public object packed_vector3_array_to_pyobject(const cpp.PackedVector3Array &p_arr)
cdef public object packed_color_array_to_pyobject(const cpp.PackedColorArray &p_arr)
cdef public object packed_vector4_array_to_pyobject(const cpp.PackedVector4Array &p_arr)
cdef public object variant_packed_byte_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_int32_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_int64_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_float32_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_float64_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_string_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_vector2_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_vector3_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_color_array_to_pyobject(const cpp.Variant &v)
cdef public object variant_packed_vector4_array_to_pyobject(const cpp.Variant &v)
cdef public void packed_byte_array_from_pyobject(object p_obj, cpp.PackedByteArray *r_ret) noexcept
cdef public void packed_int32_array_from_pyobject(object p_obj, cpp.PackedInt32Array *r_ret) noexcept
cdef public void packed_int64_array_from_pyobject(object p_obj, cpp.PackedInt64Array *r_ret) noexcept
cdef public void packed_float32_array_from_pyobject(object p_obj, cpp.PackedFloat32Array *r_ret) noexcept
cdef public void packed_float64_array_from_pyobject(object p_obj, cpp.PackedFloat64Array *r_ret) noexcept
cdef public void packed_string_array_from_pyobject(object p_obj, cpp.PackedStringArray *r_ret) noexcept
cdef public void packed_vector2_array_from_pyobject(object p_obj, cpp.PackedVector2Array *r_ret) noexcept
cdef public void packed_vector3_array_from_pyobject(object p_obj, cpp.PackedVector3Array *r_ret) noexcept
cdef public void packed_color_array_from_pyobject(object p_obj, cpp.PackedColorArray *r_ret) noexcept
cdef public void packed_vector4_array_from_pyobject(object p_obj, cpp.PackedVector4Array *r_ret) noexcept
cdef public void variant_packed_byte_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_int32_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_int64_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_float32_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_float64_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_string_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_vector2_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_vector3_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_color_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept
cdef public void variant_packed_vector4_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept

cdef public object variant_to_pyobject(const cpp.Variant &v)
cdef public void variant_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept


cdef class Pointer:
    cdef void *ptr

    @staticmethod
    cdef Pointer create(const void *ptr)

cdef Pointer pointer_to_pyobject(const void *ptr)
cdef int pointer_from_pyobject(Pointer p_obj, void **r_ret) except -1


cdef class Buffer:
    cdef uint8_t *ptr
    cdef readonly int64_t size

cdef class IntPointer(Pointer):
    pass

cdef class FloatPointer(Pointer):
    pass


cdef class _CStructDataKeeper:
    cdef numpy.ndarray _c_struct_data


cdef class AudioFrame(_CStructDataKeeper):
    cdef public float left
    cdef public float right

cdef AudioFrame audio_frame_to_pyobject(const cpp.AudioFrame *af)
cdef int audio_frame_from_pyobject(AudioFrame p_obj, cpp.AudioFrame *r_ret) except -1


cdef class CaretInfo(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public int leading_direction
    cdef public int trailing_direction

cdef CaretInfo caret_info_to_pyobject(const cpp.CaretInfo *ci)
cdef int caret_info_from_pyobject(CaretInfo p_obj, cpp.CaretInfo *r_ret) except -1


cdef class Glyph(_CStructDataKeeper):
    cdef public int start
    cdef public int end
    cdef public uint8_t count
    cdef public uint8_t repeat
    cdef public uint16_t flags
    cdef public float x_off
    cdef public float y_off
    cdef public float advance
    cdef public RID font_rid
    cdef public int font_size
    cdef public int32_t index


cdef Glyph glyph_to_pyobject(const cpp.Glyph *g)
cdef int glyph_from_pyobject(Glyph p_obj, cpp.Glyph *r_ret) except -1


cdef class ObjectID(_CStructDataKeeper):
    cdef public uint64_t id

cdef object object_id_to_pyobject(const cpp.ObjectID *oid)
cdef int object_id_from_pyobject(object p_obj, cpp.ObjectID *r_ret) except -1


cdef class PhysicsServer2DExtensionMotionResult(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public float collision_depth
    cdef public float collision_safe_fraction
    cdef public float collision_unsafe_fraction
    cdef public int collision_local_shape
    cdef public ObjectID collider_id
    cdef public RID collider
    cdef public int collider_shape

ctypedef PhysicsServer2DExtensionMotionResult _PS2DEMotionResult
cdef _PS2DEMotionResult physics_server2d_extension_motion_result_to_pyobject(const cpp._PS2DEMotionResult *)
cdef int physics_server2d_extension_motion_result_from_pyobject(_PS2DEMotionResult, cpp._PS2DEMotionResult *) except -1


cdef class PhysicsServer2DExtensionRayResult(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public RID rid
    cdef public ObjectID collider_id
    cdef public Object collider
    cdef public int shape

ctypedef PhysicsServer2DExtensionRayResult _PS2DERayResult
cdef _PS2DERayResult physics_server2d_extension_ray_result_to_pyobject(const cpp._PS2DERayResult *)
cdef int physics_server2d_extension_ray_result_from_pyobject(_PS2DERayResult, cpp._PS2DERayResult *) except -1


cdef class PhysicsServer2DExtensionShapeRestInfo(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public RID rid
    cdef public ObjectID collider_id
    cdef public int shape

ctypedef PhysicsServer2DExtensionShapeRestInfo _PS2DEShapeRestInfo
cdef _PS2DEShapeRestInfo physics_server2d_extension_shape_rest_info_to_pyobject(const cpp._PS2DEShapeRestInfo *)
cdef int physics_server2d_extension_shape_rest_info_from_pyobject(_PS2DEShapeRestInfo,
                                                                  cpp._PS2DEShapeRestInfo *) except -1


cdef class PhysicsServer2DExtensionShapeResult(_CStructDataKeeper):
    cdef public RID rid
    cdef public ObjectID collider_id
    cdef public Object collider
    cdef public int shape

ctypedef PhysicsServer2DExtensionShapeResult _PS2DEShapeResult
cdef _PS2DEShapeResult physics_server2d_extension_shape_result_to_pyobject(const cpp._PS2DEShapeResult *)
cdef int physics_server2d_extension_shape_result_from_pyobject(_PS2DEShapeResult, cpp._PS2DEShapeResult *) except -1


cdef class PhysicsServer3DExtensionMotionCollision(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public float depth
    cdef public int local_shape
    cdef public ObjectID collider_id
    cdef public RID collider
    cdef public int collider_shape

ctypedef PhysicsServer3DExtensionMotionCollision _PS3DEMotionCollision
cdef _PS3DEMotionCollision physics_server3d_extension_motion_collision_to_pyobject(const cpp._PS3DEMotionCollision *)
cdef int physics_server3d_extension_motion_collision_from_pyobject(_PS3DEMotionCollision,
                                                                   cpp._PS3DEMotionCollision *) except -1


cdef class PhysicsServer3DExtensionMotionResult(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public float collision_depth
    cdef public float collision_safe_fraction
    cdef public float collision_unsafe_fraction
    cdef public list collisions

ctypedef PhysicsServer3DExtensionMotionResult _PS3DEMotionResult
cdef _PS3DEMotionResult physics_server3d_extension_motion_result_to_pyobject(const cpp._PS3DEMotionResult *)
cdef int physics_server3d_extension_motion_result_from_pyobject(_PS3DEMotionResult, cpp._PS3DEMotionResult *) except -1


cdef class PhysicsServer3DExtensionRayResult(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public RID rid
    cdef public ObjectID collider_id
    cdef public Object collider
    cdef public int shape
    cdef public int face_index

ctypedef PhysicsServer3DExtensionRayResult _PS3DERayResult
cdef _PS3DERayResult physics_server3d_extension_ray_result_to_pyobject(const cpp._PS3DERayResult *)
cdef int physics_server3d_extension_ray_result_from_pyobject(_PS3DERayResult, cpp._PS3DERayResult *) except -1


cdef class PhysicsServer3DExtensionShapeRestInfo(_CStructDataKeeper):
    cdef numpy.ndarray data
    cdef public RID rid
    cdef public ObjectID collider_id
    cdef public int shape

ctypedef PhysicsServer3DExtensionShapeRestInfo _PS3DEShapeRestInfo
cdef _PS3DEShapeRestInfo physics_server3d_extension_shape_rest_info_to_pyobject(const cpp._PS3DEShapeRestInfo *)
cdef int physics_server3d_extension_shape_rest_info_from_pyobject(_PS3DEShapeRestInfo,
                                                                  cpp._PS3DEShapeRestInfo *) except -1

cdef class PhysicsServer3DExtensionShapeResult(_CStructDataKeeper):
    cdef public RID rid
    cdef public ObjectID collider_id
    cdef public Object collider
    cdef public int shape

ctypedef PhysicsServer3DExtensionShapeResult _PS3DEShapeResult
cdef _PS3DEShapeResult physics_server3d_extension_shape_result_to_pyobject(const cpp._PS3DEShapeResult *)
cdef int physics_server3d_extension_shape_result_from_pyobject(_PS3DEShapeResult, cpp._PS3DEShapeResult *) except -1


cdef class ScriptLanguageExtensionProfilingInfo(_CStructDataKeeper):
    cdef public str signature
    cdef public uint64_t call_count
    cdef public uint64_t total_time
    cdef public uint64_t self_time

ctypedef ScriptLanguageExtensionProfilingInfo _SLEPInfo
cdef _SLEPInfo script_language_extension_profiling_info_to_pyobject(const cpp._SLEPInfo *)
cdef int script_language_extension_profiling_info_from_pyobject(_SLEPInfo, cpp._SLEPInfo *)


cdef variant_to_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_to_pyobject_funcs
cdef variant_from_pyobject_func_t[<int>cpp.VARIANT_MAX] variant_from_pyobject_funcs

cdef cpp.VariantType pytype_to_variant_type(object p_type) noexcept
cdef cpp.VariantType pyobject_to_variant_type(object p_obj) noexcept

cdef ArgType pytype_to_argtype(object p_type) noexcept
