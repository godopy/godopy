"""\
This module wraps objects inside the engine
"""
from godot_cpp cimport *
from cpython cimport ref, PyObject

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
    cdef dict _methods
    cdef public str __name__

cdef class MethodBind:
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _gde_mb
    cdef str returning_type
    cdef Variant _ptrcall_string(self, GDExtensionConstTypePtr *p_args) noexcept nogil
    cpdef object _call_internal(self, tuple args)

cdef class UtilityFunction:
    cdef GDExtensionPtrUtilityFunction _gde_uf
    cdef str returning_type
    cpdef object _call_internal(self, tuple args)

cdef UtilityFunction __print
cdef UtilityFunction _printerr
cdef UtilityFunction _printraw
cdef UtilityFunction _print_verbose
cdef UtilityFunction _print_rich
cdef UtilityFunction _push_error
cdef UtilityFunction _push_warning
