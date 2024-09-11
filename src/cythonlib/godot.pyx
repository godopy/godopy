cimport cython
from cpython cimport ref, PyObject
from libc.stdlib cimport malloc, free
from godot_cpp cimport *

import sys
import builtins

include "type_converters.pxi"

cdef class GodotObject:
    cdef void* _owner
    cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef StringName class_name
    cdef readonly str __godot_class__

    @staticmethod
    cdef GodotObject from_ptr(void *ptr):
        cdef GodotSingleton self = GodotSingleton.__new__(GodotSingleton)
        self._owner = ptr

        return self

    @staticmethod
    cdef PyObject* _create_callback_gil(void *p_token, void *p_instance):
        print("CREATE CALLBACK", <int64_t>p_instance)
        cdef GodotSingleton wrapper = GodotObject.from_ptr(p_instance)
        ref.Py_INCREF(wrapper)

        print("CREATED BINDING", <int64_t><PyObject *>wrapper)
        return <PyObject *>wrapper

    @staticmethod
    cdef void _free_callback_gil(void *p_token, void *p_instance, void *p_binding):
        print("FREE CALLBACK", <int64_t>p_instance, <int64_t>p_binding)
        cdef GodotSingleton wrapper = <object>p_binding
        ref.Py_DECREF(wrapper)

    @staticmethod
    cdef void* _create_callback(void *p_token, void *p_instance) noexcept nogil:
        with gil:
            return <void *>GodotSingleton._create_callback_gil(p_token, p_instance)

    @staticmethod
    cdef void _free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil:
        with gil:
            GodotSingleton._free_callback_gil(p_token, p_instance, p_binding)

    @staticmethod
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance, GDExtensionBool p_reference) noexcept nogil:
        return True

    def __cinit__(self):
        self._binding_callbacks.create_callback = &GodotObject._create_callback
        self._binding_callbacks.free_callback = &GodotObject._free_callback
        self._binding_callbacks.reference_callback = &GodotObject._reference_callback
    
    def __init__(self, str class_name):
        self.__godot_class__ = class_name
        self.class_name = stringname_from_str(class_name)
        self._owner = gdextension_interface_classdb_construct_object(&self.class_name)
        # ???  gdextension_interface_object_set_instance(self._owner, snn_from_str(class_name))
        gdextension_interface_object_set_instance_binding(self._owner, &self.class_name, <void *><PyObject *>self, &self._binding_callbacks)


cdef class GodotSingleton(GodotObject):
    # cdef void* owner
    # cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef GDExtensionObjectPtr singleton_obj
    cdef void* singleton

    def __init__(self, str class_name):
        self.__godot_class__ = class_name
        self.class_name = stringname_from_str(class_name)
        self.singleton_obj = gdextension_interface_global_get_singleton(&self.class_name)
        self.singleton = gdextension_interface_object_get_instance_binding(self.singleton_obj, token, &self._binding_callbacks)


cdef class GodotMethodBindRet:
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _mb
    cdef StringName method_name

    def __cinit__(self, GodotObject wrapper, str method_name, GDExtensionInt method_hash):
        self._owner = wrapper._owner
        self.method_name = stringname_from_str(method_name)
        self._mb = gdextension_interface_classdb_get_method_bind(&wrapper.class_name, &self.method_name, method_hash)

    cpdef object _call_internal(self, tuple args):
        cdef Variant ret
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>malloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg
        
        with nogil:
            gdextension_interface_object_method_bind_ptrcall(self._mb, self._owner, p_args, &ret)
            free(p_args)

        return pyobject_from_variant(ret)
    
    def __call__(self, *args):
        return self._call_internal(args)


cdef class GodotMethodBindNoRet(GodotMethodBindRet):
    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>malloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            gdextension_interface_object_method_bind_ptrcall(self._mb, self._owner, p_args, NULL)
            free(p_args)


# cpdef printraw(*args):
#    cdef GDExtensionPtrUtilityFunction f = gdextension_interface_variant_get_ptr_utility_function(StringName(b"printraw")._native_ptr(), 2648703342)

cdef class GodotUtilityFunctionRet:
    cdef GDExtensionPtrUtilityFunction _uf
    cdef StringName function_name

    def __cinit__(self, str function_name, GDExtensionInt function_hash):
        self.function_name = stringname_from_str(function_name)
        self._uf = gdextension_interface_variant_get_ptr_utility_function(&self.function_name, function_hash)
    
    cpdef object _call_internal(self, tuple args):
        cdef Variant ret
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>malloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        cdef int size = len(args)
        for i in range(size):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg
        
        with nogil:
            self._uf(&ret, p_args, size)
            free(p_args)

        return pyobject_from_variant(ret)

    def __call__(self, *args):
        return self._call_internal(args)


cdef class GodotUtilityFunctionNoRet(GodotUtilityFunctionRet):
    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>malloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        cdef int size = len(args)
        for i in range(size):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg
        
        with nogil:
            self._uf(NULL, p_args, size)
            free(p_args)

printraw = GodotUtilityFunctionNoRet('printraw', 2648703342)
print_rich = GodotUtilityFunctionNoRet('print_rich', 2648703342)
push_error = GodotUtilityFunctionNoRet('push_error', 2648703342)

include "sys_util.pxi"
