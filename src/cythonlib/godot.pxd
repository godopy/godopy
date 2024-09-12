from godot_cpp cimport *
from cpython cimport PyObject

cdef class GodotObject:
    cdef void* _owner
    cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef StringName _class_name
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
    cdef str name
    cdef StringName _name


cdef class GodotSingletonClass(GodotClass):
    pass
