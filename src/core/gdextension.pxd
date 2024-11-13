from libc.stdint cimport int8_t
from cpython cimport PyObject

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy

from binding cimport *
from godot_cpp cimport *

from godot_types cimport (
    String as PyGDString,
    StringName as PyGDStringName,
)

from typing import List

Str = str | PyGDString | PyGDStringName


cdef int configure(object config) except -1


cpdef str variant_type_to_str(VariantType vartype)
cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX


cdef public object object_to_pyobject(void *p_godot_object)
cdef public object variant_object_to_pyobject(const Variant &v)
cdef public void object_from_pyobject(object p_obj, void **r_ret) noexcept
cdef public void cppobject_from_pyobject(object p_obj, GodotCppObject **r_ret) noexcept
cdef public void variant_object_from_pyobject(object p_obj, Variant *r_ret) noexcept

cdef object script_instance_to_pyobject(void *)
cdef int script_instance_from_pyobject(ScriptInstance p_obj, void **) except -1

cdef public object callable_to_pyobject(const GodotCppCallable &p_callable)
cdef public object variant_callable_to_pyobject(const Variant &v)
cdef public void callable_from_pyobject(object p_obj, GodotCppCallable *r_ret) noexcept
cdef public void variant_callable_from_pyobject(object p_obj, Variant *r_ret) noexcept


cdef dict _NODEDB
cdef dict _OBJECTDB
cdef dict _METHODDB
cdef dict _BUILTIN_METHODDB
cdef dict _CLASSDB
cdef dict _bound_method_cache
cdef dict ALLOCATIONS


cdef enum ArgType:
    ARGTYPE_NIL = NIL
    ARGTYPE_BOOL
    ARGTYPE_INT
    ARGTYPE_FLOAT
    ARGTYPE_STRING
    ARGTYPE_VECTOR2
    ARGTYPE_VECTOR2I
    ARGTYPE_RECT2
    ARGTYPE_RECT2I
    ARGTYPE_VECTOR3
    ARGTYPE_VECTOR3I
    ARGTYPE_TRANSFORM2D
    ARGTYPE_VECTOR4
    ARGTYPE_VECTOR4I
    ARGTYPE_PLANE
    ARGTYPE_QUATERNION
    ARGTYPE_AABB
    ARGTYPE_BASIS
    ARGTYPE_TRANSFORM3D
    ARGTYPE_PROJECTION
    ARGTYPE_COLOR
    ARGTYPE_STRING_NAME
    ARGTYPE_NODE_PATH
    ARGTYPE_RID
    ARGTYPE_OBJECT
    ARGTYPE_CALLABLE
    ARGTYPE_SIGNAL
    ARGTYPE_DICTIONARY
    ARGTYPE_ARRAY
    ARGTYPE_PACKED_BYTE_ARRAY
    ARGTYPE_PACKED_INT32_ARRAY
    ARGTYPE_PACKED_INT64_ARRAY
    ARGTYPE_PACKED_FLOAT32_ARRAY
    ARGTYPE_PACKED_FLOAT64_ARRAY
    ARGTYPE_PACKED_STRING_ARRAY
    ARGTYPE_PACKED_VECTOR2_ARRAY
    ARGTYPE_PACKED_VECTOR3_ARRAY
    ARGTYPE_PACKED_COLOR_ARRAY
    ARGTYPE_PACKED_VECTOR4_ARRAY

    ARGTYPE_VARIANT
    ARGTYPE_POINTER
    ARGTYPE_SCRIPT_INSTANCE

    ARGTYPE_AUDIO_FRAME
    ARGTYPE_CARET_INFO
    ARGTYPE_GLYPH
    ARGTYPE_OBJECT_ID

    ARGTYPE_PHYSICS_SERVER2D_MOTION_RESULT
    ARGTYPE_PHYSICS_SERVER2D_RAY_RESULT
    ARGTYPE_PHYSICS_SERVER2D_SHAPE_REST_INFO
    ARGTYPE_PHYSICS_SERVER2D_SHAPE_RESULT
    ARGTYPE_PHYSICS_SERVER3D_MOTION_COLLISION
    ARGTYPE_PHYSICS_SERVER3D_MOTION_RESULT
    ARGTYPE_PHYSICS_SERVER3D_RAY_RESULT
    ARGTYPE_PHYSICS_SERVER3D_SHAPE_REST_INFO
    ARGTYPE_PHYSICS_SERVER3D_SHAPE_RESULT
    ARGTYPE_SCRIPTING_LANGUAGE_PROFILING_INFO

    ARGTYPE_MAX
    ARGTYPE_NO_ARGTYPE = -1


cdef class _Memory:
    cdef void *ptr
    cdef size_t num_bytes

    cdef void *realloc(self, size_t p_bytes) except NULL nogil
    cdef void free(self) noexcept nogil


cdef public class Object [object GDPyObject, type GDPyObject_Type]:
    cdef void *_owner
    cdef void *_ref_owner
    cdef bint _instance_set
    cdef bint _needs_cleanup

    cdef readonly bint is_singleton
    cdef readonly Class __godot_class__


cdef class VariantMethod:
    cdef readonly str __name__
    cdef object __self__
    cdef Variant _self_variant
    cdef StringName _method

    cdef void _varcall(self, const Variant **p_args, size_t size, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil


cdef class VariantStaticMethod:
    cdef readonly str __name__
    cdef readonly VariantType __self__
    cdef StringName _method

    cdef void _varcall(self, const Variant **p_args, size_t size, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil


cdef class MethodBind:
    cdef readonly str __name__
    cdef readonly tuple type_info
    cdef int8_t[16] _type_info_opt

    cdef readonly bint is_vararg

    cdef readonly Object __self__
    cdef void *_self_owner
    cdef GDExtensionMethodBindPtr _godot_method_bind
    cdef object key
    cdef object func

    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil
    cdef void _varcall(self, const Variant **p_args, size_t size, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil


cdef class ScriptMethod:
    cdef readonly str __name__
    cdef readonly Object __self__
    cdef void *_self_owner
    cdef StringName _method

    cdef void _varcall(self, const Variant **p_args, size_t size, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil


cdef class UtilityFunction:
    cdef readonly str __name__
    cdef readonly tuple type_info
    cdef int8_t[16] _type_info_opt

    cdef GDExtensionPtrUtilityFunction _godot_utility_function

    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil


cdef class BuiltinMethod:
    cdef readonly str __name__
    cdef readonly tuple type_info
    cdef int8_t[16] _type_info_opt

    cdef object __self__
    cdef void *_self_owner
    cdef GDExtensionPtrBuiltInMethod _godot_builtin_method

    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil

    @staticmethod
    cdef BuiltinMethod new_with_selfptr(object owner, object method_name, void *selfptr)


cdef class PropertyInfo:
    cdef public VariantType type
    cdef public object name
    cdef public object class_name
    cdef public uint32_t hint
    cdef public object hint_string
    cdef public uint32_t usage


cdef class MethodInfo:
    cdef public object name
    cdef public PropertyInfo return_value
    cdef public uint32_t flags
    cdef public int32_t id
    cdef public list arguments
    cdef public list default_arguments


cdef class Callable:
    # TODO: Try to use GDExtension API directly without godot-cpp objects
    cdef GodotCppCallable _godot_callable

    @staticmethod
    cdef Callable from_cpp(const GodotCppCallable &)


cdef class PythonCallable(Callable):
    cdef readonly object __name__
    cdef readonly object __func__
    cdef readonly object __self__

    cdef readonly object type_info
    cdef int8_t[16] _type_info_opt

    cdef readonly size_t error_count
    cdef readonly bint initialized

    @staticmethod
    cdef void call_callback(void *callable_userdata, const (const void *) *p_args, int64_t p_count, void *r_return,
                            GDExtensionCallError *r_error) noexcept nogil

    @staticmethod
    cdef uint32_t hash_callback(void *callable_userdata) noexcept nogil

    cdef uint32_t hash(self) except -1

    @staticmethod
    cdef uint8_t equal_callback(void *callable_userdata_a, void *callable_userdata_b) noexcept nogil

    @staticmethod
    cdef uint8_t less_than_callback(void *callable_userdata_a, void *callable_userdata_b) noexcept nogil

    @staticmethod
    cdef void to_string_callback(void *callable_userdata, uint8_t *r_is_valid, void *r_out) noexcept nogil

    @staticmethod
    cdef int64_t get_argument_count_callback(void *callable_userdata, uint8_t *r_is_valid) noexcept nogil

    cdef int64_t get_argument_count(self) except -1

    @staticmethod
    cdef void free_callback(void *callable_userdata) noexcept nogil

    cdef int free(self) except -1


cdef class _PropertyInfoDataArray:
    cdef _Memory memory
    cdef size_t count
    cdef list names
    cdef list classnames
    cdef list hintstrings

    cdef GDExtensionPropertyInfo *ptr(self) noexcept nogil


cdef class _MethodInfoDataArray:
    cdef _Memory memory
    cdef size_t count
    cdef list names
    cdef _PropertyInfoDataArray return_values
    cdef list arguments

    cdef GDExtensionMethodInfo *ptr(self) noexcept nogil


cdef class ScriptInstance:
    cdef void *_godot_script_instance
    cdef _Memory _info

    cdef readonly object __name__
    cdef readonly Extension __language__
    cdef readonly Extension __script__
    cdef readonly Object __owner__
    cdef readonly dict __script_dict__

    cdef _PropertyInfoDataArray property_info_data
    cdef _MethodInfoDataArray method_info_data

    @staticmethod
    cdef uint8_t set_callback(void *p_instance, const void *p_name, const void *p_value) noexcept nogil

    @staticmethod
    cdef uint8_t get_callback(void *p_instance, const void *p_name, void *r_ret) noexcept nogil

    @staticmethod
    cdef const GDExtensionPropertyInfo *get_property_list_callback(void *p_instance, uint32_t *r_count) noexcept nogil

    @staticmethod
    cdef void *get_owner_callback(void *p_instance) noexcept nogil

    @staticmethod
    cdef const GDExtensionMethodInfo *get_method_list_callback(void *p_instance, uint32_t *r_count) noexcept nogil

    @staticmethod
    cdef uint8_t has_method_callback(void *p_instance, const void *p_name) noexcept nogil

    @staticmethod
    cdef int64_t get_method_argument_count_callback(void *p_instance, const void *p_name, uint8_t *r_is_valid) noexcept nogil

    @staticmethod
    cdef void call_callback(void *p_instance, const void *p_method, const (const void *) *p_args, int64_t p_count,
                            void *r_ret, GDExtensionCallError *r_error) noexcept nogil

    @staticmethod
    cdef void notification_callback(void *p_instance, int32_t p_what, uint8_t p_reversed) noexcept nogil

    @staticmethod
    cdef void to_string_callback(void *p_instance, uint8_t *r_is_valid, void *r_out) noexcept nogil

    @staticmethod
    cdef void *get_script_callback(void *p_instance) noexcept nogil

    @staticmethod
    cdef uint8_t is_placeholder_callback(void *p_instance) noexcept nogil

    @staticmethod
    cdef void *get_language_callback(void *p_instance) noexcept nogil

    @staticmethod
    cdef void free_callback(void *p_instance) noexcept nogil

    cdef int free(self) except -1


cdef class Class:
    cdef readonly dict __method_info__
    cdef readonly str __name__
    cdef readonly Class __inherits__
    cdef void *_godot_class_tag

    cdef int initialize_class(self) except -1
    cpdef object get_method_info(self, method_name)

    cdef void *get_tag(self) except NULL

    @staticmethod
    cdef Class get_class(object name)


cdef class _ExtensionMethodBase:
    cdef readonly str __name__
    cdef readonly object __func__
    cdef readonly bint is_registered
    cdef readonly tuple type_info

    cdef list get_default_arguments(self)
    cdef PropertyInfo get_argument_info(self, int pos)
    cdef PropertyInfo get_return_info(self)
    cdef list get_argument_info_list(self)
    cdef int get_return_metadata(self) noexcept
    cdef int metadata_from_type(self, VariantType t) noexcept nogil
    cdef list get_argument_metadata_list(self)
    cdef GDExtensionBool has_return(self) noexcept
    cdef uint32_t get_argument_count(self) noexcept


cdef class ExtensionVirtualMethod(_ExtensionMethodBase):
    cdef int register(self, ExtensionClass cls) except -1


cdef class ExtensionMethod(_ExtensionMethodBase):
    cdef int register(self, ExtensionClass cls) except -1

    @staticmethod
    cdef void call_callback(void *p_method_userdata, void *p_instance, const (const void *) *p_args, int64_t p_count,
                            void *r_return, GDExtensionCallError *r_error) noexcept nogil

    cdef int call(self, object instance, const Variant **p_args, size_t p_count, Variant *r_ret,
                  GDExtensionCallError *r_error) except -1

    @staticmethod
    cdef void ptrcall_callback(void *p_method_userdata, void *p_instance, const (const void *) *p_args,
                               void *r_return) noexcept nogil

    cdef int ptrcall(self, object instance, const void **p_args, void *r_ret) except -1


cdef class ExtensionEnum:
    cdef readonly object __name__
    cdef readonly object __enum__
    cdef readonly bint is_registered
    cdef readonly bint is_bitfield

    cdef int register(self, ExtensionClass cls) except -1


cdef class ExtensionProperty:
    cdef readonly object __name__
    cdef readonly object __setter_name__
    cdef readonly object __getter_name__
    cdef readonly int64_t __idx__
    cdef readonly PropertyInfo __info__
    cdef readonly bint is_registered
    cdef readonly bint is_indexed

    cdef int register(self, ExtensionClass cls) except -1


cdef class ExtensionPropertyGroup:
    cdef readonly object __name__
    cdef readonly object __prefix__
    cdef readonly bint is_registered
    cdef readonly bint is_subgroup

    cdef int register(self, ExtensionClass cls) except -1


cdef class ExtensionSignal:
    cdef readonly object __name__
    cdef readonly object __arguments__
    cdef readonly bint is_registered

    cdef int register(self, ExtensionClass cls) except -1


cdef class ExtensionClassBindings:
    cdef readonly dict method
    cdef readonly dict pymethod
    cdef readonly dict ownvirtual
    cdef readonly dict gdvirtual
    cdef readonly dict intenum
    cdef readonly dict bitfield
    cdef readonly dict prop
    cdef readonly dict idxprop
    cdef readonly dict group
    cdef readonly dict subgroup
    cdef readonly dict signal


cdef class ExtensionClass(Class):
    cdef readonly bint is_registered
    cdef readonly bint is_virtual
    cdef readonly bint is_abstract
    cdef readonly bint is_exposed
    cdef readonly bint is_runtime

    cdef readonly ExtensionClassBindings _bind

    cdef list _used_refs

    cdef object get_method_and_method_type_info(self, object name)
    cdef void *get_method_and_method_type_info_ptr(self, object name) except NULL
    cdef void *get_special_method_info_ptr(self, SpecialMethod method) except NULL

    cdef int register_method(self, func: typing.Callable, name: Str) except -1
    cdef int register_virtual_method(self, func: typing.Callable, name: Str) except -1
    cdef int register_enum(self, enum_obj: enum.IntEnum, bint is_bitfield=*) except -1
    cdef int register_property(self, PropertyInfo info, setter_name: Str, getter_name: Str) except -1
    cdef int register_property_indexed(self, PropertyInfo info, setter_name: Str, getter_name: Str,
                                       int64_t index) except -1
    cdef int register_property_group(self, group_name: Str, prefix: Str) except -1
    cdef int register_property_subgroup(self, subgroup_name: Str, prefix: Str) except -1
    cdef int register_signal(self, signal_name: Str, arguments: List[PropertyInfo]) except -1

    cdef int set_registered(self) except -1
    cdef int unregister(self) except -1

    @staticmethod
    cdef void *create_instance_callback(void *p_class_userdata, uint8_t p_notify) noexcept nogil

    @staticmethod
    cdef void free_instance_callback(void *p_class_userdata, void *p_instance) noexcept nogil

    @staticmethod
    cdef void *recreate_instance_callback(void *p_data, void *p_instance) noexcept nogil

    @staticmethod
    cdef void *get_virtual_call_data_callback(void *p_userdata, void *p_name) noexcept nogil

    cdef int free_instance(self, object instance) except -1
    cdef void *create_instance(self, bint p_notify) except? NULL
    cdef int get_virtual_call_data(self, object name, void **r_ret) except -1

    cdef int _register(self, object kwargs) except -1


cdef enum SpecialMethod:
    _THREAD_ENTER = 1
    _THREAD_EXIT = 2
    _FRAME = 3


cdef public class Extension(Object) [object GDPyExtension, type GDPyExtension_Type]:
    cdef dict _callable_cache
    cdef _PropertyInfoDataArray property_info_data

    @staticmethod
    cdef uint8_t set_callback(void *p_instance, const void *p_name, const void *p_value) noexcept nogil

    @staticmethod
    cdef uint8_t get_callback(void *p_instance, const void *p_name, void *r_ret) noexcept nogil

    @staticmethod
    cdef const GDExtensionPropertyInfo *get_property_list_callback(void *p_instance, uint32_t *r_count) noexcept nogil

    @staticmethod
    cdef void notification_callback(void *p_instance, int32_t p_what, uint8_t p_reversed) noexcept nogil

    @staticmethod
    cdef void to_string_callback(void *p_instance, uint8_t *r_is_valid, void *r_out) noexcept nogil

    @staticmethod
    cdef void call_virtual_with_data_callback(void *p_instance, const void *p_name, void *p_func,
                                              const (const void *) *p_args, void *r_ret) noexcept nogil

    cdef void call_special_virtual(self, SpecialMethod method) noexcept nogil
    cdef int call_virtual_with_data(self, object func_and_info, const void **p_args, void *r_ret) except -1
