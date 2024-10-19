"""\
This module wraps objects inside the engine
"""
from binding cimport *
from godot_cpp cimport *
from cpython cimport PyObject
cimport numpy


cpdef str variant_type_to_str(VariantType vartype)
cpdef VariantType str_to_variant_type(str vartype) except VARIANT_MAX


cdef dict _NODEDB
cdef dict _OBJECTDB
cdef dict _METHODDB
cdef dict _CLASSDB
cdef list _registered_classes


cdef public class Object [object GDPyObject, type GDPyObject_Type]:
    cdef void *_owner
    cdef void *_ref_owner  # According to gdextension_interface.h, if _owner is Ref, this would be real owner
    cdef bint is_singleton
    cdef readonly Class __godot_class__


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
                                      GDExtensionTypePtr r_ret) noexcept with gil


cdef class Class:
    cdef readonly dict __method_info__
    cdef readonly str __name__
    cdef readonly Class __inherits__

    cdef int initialize_class(self) except -1
    cpdef object get_method_info(self, method_name)

    @staticmethod
    cdef Class get_class(str name)


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


cdef class _CallableBase:
    cdef str __name__
    cdef tuple type_info

    cpdef object _call_internal(self, tuple args)
    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil


cdef class MethodBind(_CallableBase):
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _godot_method_bind
    cdef Object __owner__

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil


cdef class UtilityFunction(_CallableBase):
    cdef GDExtensionPtrUtilityFunction _godot_utility_function

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil
