"""\
This module wraps objects inside the engine
"""
from binding cimport *
from godot_cpp cimport *
from cpython cimport PyObject

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy


cpdef str variant_type_to_str(VariantType vartype)
cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX

cdef public object object_to_pyobject(void *p_godot_object)
cdef public object variant_object_to_pyobject(const Variant &v)
cdef public void object_from_pyobject(object p_obj, void **r_ret) noexcept
cdef public void variant_object_from_pyobject(object p_obj, Variant *r_ret) noexcept


cdef dict _NODEDB
cdef dict _OBJECTDB
cdef dict _METHODDB
cdef dict _BUILTIN_METHODDB
cdef dict _CLASSDB
cdef list _registered_classes


cdef class Class:
    cdef readonly dict __method_info__
    cdef readonly str __name__
    cdef readonly Class __inherits__

    cdef int initialize_class(self) except -1
    cpdef object get_method_info(self, method_name)

    @staticmethod
    cdef Class get_class(object name)


cdef public class Object [object GDPyObject, type GDPyObject_Type]:
    cdef void *_owner
    cdef void *_ref_owner  # According to gdextension_interface.h, if _owner is Ref, this would be real owner
    cdef bint is_singleton
    cdef readonly Class __godot_class__


cdef class EngineCallableBase:
    cdef readonly str __name__
    cdef readonly tuple type_info


cdef class MethodBind(EngineCallableBase):
    cdef readonly bint is_vararg

    cdef Object __self__
    cdef void *_base
    cdef GDExtensionMethodBindPtr _godot_method_bind

    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil
    cdef void _varcall(self, const Variant **p_args, size_t size, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil


cdef class UtilityFunction(EngineCallableBase):
    cdef GDExtensionPtrUtilityFunction _godot_utility_function

    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil


cdef class BuiltinMethod(EngineCallableBase):
    cdef object __self__
    cdef void *_base
    cdef GDExtensionPtrBuiltInMethod _godot_builtin_method

    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil

    @staticmethod
    cdef BuiltinMethod new_with_baseptr(object owner, object method_name, void *_base)


cdef enum SpecialMethod:
    _THREAD_ENTER = 1
    _THREAD_EXIT = 2
    _FRAME = 3


cdef class ExtensionClass(Class):
    cdef readonly bint is_registered
    cdef readonly dict method_bindings
    cdef readonly dict python_method_bindings
    cdef readonly dict virtual_method_bindings
    cdef readonly dict virtual_method_implementation_bindings

    cdef list _used_refs

    cdef tuple get_method_and_method_type_info(self, str name)
    cdef void *get_method_and_method_type_info_ptr(self, str name) except NULL
    cdef void *get_special_method_info_ptr(self, SpecialMethod method) except NULL
    cdef int set_registered(self) except -1
    cdef int unregister(self) except -1

    @staticmethod
    cdef void free_instance(void *data, void *p_instance) noexcept nogil

    @staticmethod
    cdef int _free_instance(void *p_self, void *p_instance) except -1 with gil

    @staticmethod
    cdef GDExtensionObjectPtr create_instance(void *p_class_userdata, GDExtensionBool p_notify) noexcept nogil

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance(void *p_self, bint p_notify) except? NULL with gil

    @staticmethod
    cdef GDExtensionObjectPtr recreate_instance(void *p_data, GDExtensionObjectPtr p_instance) noexcept nogil



cdef public class Extension(Object) [object GDPyExtension, type GDPyExtension_Type]:
    cdef bint _needs_cleanup

    cpdef destroy(self)

    @staticmethod
    cdef void *get_virtual_call_data(void *p_userdata, GDExtensionConstStringNamePtr p_name) noexcept nogil

    @staticmethod
    cdef void *_get_virtual_call_data(void *p_cls, const StringName &p_name) noexcept with gil

    @staticmethod
    cdef void _call_special_virtual(SpecialMethod placeholder) noexcept nogil

    @staticmethod
    cdef void call_virtual_with_data(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name,
                                     void *p_func, const GDExtensionConstTypePtr *p_args,
                                     GDExtensionTypePtr r_ret) noexcept nogil

    @staticmethod
    cdef void _call_virtual_with_data(void *p_self, void *p_func_and_info, const void **p_args,
                                      void *r_ret) noexcept with gil


cdef class PropertyInfo:
    cdef public VariantType type
    cdef public str name
    cdef public str class_name
    cdef public uint32_t hint
    cdef public str hint_string
    cdef public uint32_t usage


cdef class _ExtensionMethodBase:
    cdef readonly str __name__
    cdef readonly object __func__
    cdef readonly bint is_registered
    cdef readonly tuple type_info

    cdef list get_default_arguments(self)
    cdef PropertyInfo get_argument_info(self, int pos)
    cdef PropertyInfo get_return_info(self)
    cdef list get_argument_info_list(self)
    cdef int get_return_metadata(self) except -1
    cdef int metadata_from_type(self, VariantType t) except -1 nogil
    cdef list get_argument_metadata_list(self)
    cdef GDExtensionBool has_return(self) except -1
    cdef uint32_t get_argument_count(self) except -1
    cdef uint32_t get_default_argument_count(self) except -1


cdef class ExtensionVirtualMethod(_ExtensionMethodBase):
    cdef int register(self, ExtensionClass cls) except -1


cdef class ExtensionMethod(_ExtensionMethodBase):
    cdef int register(self, ExtensionClass cls) except -1

    @staticmethod
    cdef void call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                   const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count,
                   GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) noexcept nogil

    @staticmethod
    cdef void _call(void *p_method, void *p_self, const Variant **p_args, size_t p_argcount,
                    Variant *r_ret, GDExtensionCallError *r_error) noexcept with gil

    @staticmethod
    cdef void ptrcall(void *p_method_userdata, GDExtensionClassInstancePtr p_instance,
                      const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) noexcept nogil

    @staticmethod
    cdef void _ptrcall(void *p_method, void *p_self, const void **p_args, void *r_return) noexcept with gil


cdef class PythonCallableBase:
    cdef readonly str __name__
    cdef readonly object __func__
    cdef readonly tuple type_info


cdef class BoundExtensionMethod(PythonCallableBase):
    cdef readonly Extension __self__

    cdef size_t get_argument_count(self) except -2
