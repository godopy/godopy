"""\
This module wraps objects inside the engine
"""
from cpp cimport *
from cpython cimport ref, PyObject


cpdef str variant_to_str(VariantType vartype)
cpdef VariantType str_to_variant(str vartype)


cdef class Object:
    cdef void *_owner
    cdef bint is_singleton
    cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef readonly Class __godot_class__

    @staticmethod
    cdef Object from_ptr(void *ptr)

    @staticmethod
    cdef PyObject* _create_callback_gil(void *p_token, void *p_instance)

    @staticmethod
    cdef void _free_callback_gil(void *p_binding)

    @staticmethod
    cdef void* _create_callback(void *p_token, void *p_instance) noexcept nogil

    @staticmethod
    cdef void _free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil

    @staticmethod
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance,
                                             GDExtensionBool p_reference) noexcept nogil

cdef class Class:
    cdef readonly dict __method_info__
    cdef readonly str __name__
    cdef readonly Class __inherits__

    cdef int initialize_class(self, dict opts) except -1

    @staticmethod
    cdef Class get_class(str name, dict opts)


cdef class Callable:
    cdef tuple type_info

    cpdef object _call_internal(self, tuple args)
    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil

    # cdef int _ptrcall_string(self, Variant *r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) except -1 nogil


cdef class MethodBind(Callable):
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _godot_method_bind

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil


cdef class UtilityFunction(Callable):
    cdef GDExtensionPtrUtilityFunction _godot_utility_function

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil
