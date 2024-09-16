"""\
This module wraps objects inside the engine
"""
from godot_cpp cimport *
from cpython cimport ref, PyObject

cdef class Object:
    cdef void *_owner
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

cdef class Singleton(Object):
    pass

cdef class Class:
    cdef str __name__

cdef class SingletonClass(Class):
    pass

cdef class MethodBindRet:
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _gde_mb
    cdef str returning_type
    cdef Variant _ptrcall_string(self, GDExtensionConstTypePtr *p_args) noexcept nogil
    cpdef object _call_internal(self, tuple args)

cdef class MethodBindNoRet(MethodBindRet):
    pass

cdef class UtilityFunctionRet:
    cdef GDExtensionPtrUtilityFunction _gde_uf
    cpdef object _call_internal(self, tuple args)

cdef class UtilityFunctionNoRet(UtilityFunctionRet):
    pass

cdef UtilityFunctionNoRet _printraw
cdef UtilityFunctionNoRet _print_rich
cdef UtilityFunctionNoRet _push_error
cdef UtilityFunctionNoRet _push_warning
