"""\
This module wraps objects inside the engine
"""
from binding cimport *
from godot_cpp cimport *
from cpython cimport PyObject

cpdef str variant_type_to_str(VariantType vartype)
cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX


cdef public class Object [object GDPy_Object, type GDPy_ObjectType]:
    cdef void *_owner
    cdef void *_ref_owner  # According to gdextension_interface.h, if _owner is Ref, this would be real owner
    cdef bint is_singleton
    cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef readonly Class __godot_class__

    @staticmethod
    cdef Object from_ptr(void *ptr)

    @staticmethod
    cdef void* create_callback(void *p_token, void *p_instance) noexcept nogil

    @staticmethod
    cdef PyObject *_create_callback(void *p_owner) except NULL with gil

    @staticmethod
    cdef void free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil

    @staticmethod
    cdef void _free_callback(void *p_self) noexcept with gil

    @staticmethod
    cdef GDExtensionBool reference_callback(void *p_token, void *p_instance, GDExtensionBool p_ref) noexcept nogil


cdef public class Extension(Object) [object GDPy_Extension, type GDPy_ExtensionType]:
    cdef StringName _godot_class_name
    cdef StringName _godot_base_class_name
    cdef bint _needs_cleanup

    cpdef destroy(self)

    @staticmethod
    cdef void *get_virtual_call_data(void *p_userdata, GDExtensionConstStringNamePtr p_name) noexcept nogil

    @staticmethod
    cdef void *_get_virtual_call_data(void *p_cls, const StringName &p_name) noexcept with gil

    @staticmethod
    cdef void call_virtual_with_data(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name,
                                     void *p_func, const GDExtensionConstTypePtr *p_args,
                                     GDExtensionTypePtr r_ret) noexcept nogil

    @staticmethod
    cdef void _call_virtual_with_data(void *p_self, void *p_func_and_info, const void **p_args,
                                      GDExtensionTypePtr r_ret) noexcept with gil


cdef class Class:
    cdef readonly dict __method_info__
    cdef readonly str __name__
    cdef readonly Class __inherits__

    cdef int initialize_class(self) except -1
    cpdef object get_method_info(self, method_name)

    @staticmethod
    cdef Class get_class(str name)


cdef class ExtensionClass(Class):
    cdef readonly bint is_registered
    cdef readonly dict method_bindings
    cdef readonly dict python_method_bindings
    cdef readonly dict virtual_method_bindings
    cdef readonly dict virtual_method_implementation_bindings

    cdef list _used_refs

    cdef object get_method_and_method_type_info(self, str name)
    cdef void *get_method_and_method_type_info_ptr(self, str name) except NULL
    cdef void set_registered(self) noexcept nogil

    @staticmethod
    cdef void free_instance(void *data, void *p_instance) noexcept nogil

    @staticmethod
    cdef void _free_instance(void *p_self, void *p_instance) noexcept with gil

    @staticmethod
    cdef GDExtensionObjectPtr create_instance(void *p_class_userdata, GDExtensionBool p_notify) noexcept nogil

    @staticmethod
    cdef GDExtensionObjectPtr _create_instance(void *p_self, bint p_notify) except? NULL with gil

    @staticmethod
    cdef GDExtensionObjectPtr recreate_instance(void *p_data, GDExtensionObjectPtr p_instance) noexcept nogil


cdef class _CallableBase:
    cdef tuple type_info

    cpdef object _call_internal(self, tuple args)
    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil


cdef class MethodBind(_CallableBase):
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _godot_method_bind

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil


cdef class UtilityFunction(_CallableBase):
    cdef GDExtensionPtrUtilityFunction _godot_utility_function

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil
