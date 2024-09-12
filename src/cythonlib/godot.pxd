from godot_cpp cimport *
from cpython cimport PyObject

cdef class GodotObject:
    cdef void *_owner
    cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef readonly str __godot_class__

    @staticmethod
    cdef GodotObject from_ptr(void *ptr)

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

cdef class GodotSingleton(GodotObject):
    cdef GDExtensionObjectPtr _gde_so
    cdef void* singleton

cdef class GodotClass:
    cdef str __name__

cdef class GodotSingletonClass(GodotClass):
    pass

cdef class GodotMethodBindRet:
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _gde_mb
    cpdef object _call_internal(self, tuple args)

cdef class GodotMethodBindNoRet(GodotMethodBindRet):
    pass

cdef class GodotUtilityFunctionRet:
    cdef GDExtensionPtrUtilityFunction _gde_uf
    cpdef object _call_internal(self, tuple args)

cdef class GodotUtilityFunctionNoRet(GodotUtilityFunctionRet):
    pass

cdef GodotUtilityFunctionNoRet printraw
cdef GodotUtilityFunctionNoRet print_rich
cdef GodotUtilityFunctionNoRet push_error
