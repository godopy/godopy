cimport cython
from cpython cimport ref, PyObject
from godot cimport *

import sys
import builtins

include "gde_shortcuts.pxi"
include "type_converters.pxi"

cdef class GodotObject:
    cdef void* _owner
    cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef StringName _class_name
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
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance,
                                             GDExtensionBool p_reference) noexcept nogil:
        return True

    def __cinit__(self):
        self._binding_callbacks.create_callback = &GodotObject._create_callback
        self._binding_callbacks.free_callback = &GodotObject._free_callback
        self._binding_callbacks.reference_callback = &GodotObject._reference_callback
    
    def __init__(self, str class_name):
        self.__godot_class__ = class_name
        self._class_name = stringname_from_str(class_name)
        self._owner = _gde_classdb_construct_object(self._class_name._native_ptr())
        # ???  _gde_object_set_instance(self._owner, snn_from_str(class_name))
        _gde_object_set_instance_binding(self._owner,
                                         self._class_name._native_ptr(),
                                         <void *><PyObject *>self, &self._binding_callbacks)


cdef class GodotSingleton(GodotObject):
    # cdef void* owner
    # cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    cdef GDExtensionObjectPtr _gde_so
    cdef void* singleton

    def __init__(self, str class_name):
        self.__godot_class__ = class_name
        self._class_name = stringname_from_str(class_name)
        self._gde_so = _gde_global_get_singleton(self._class_name._native_ptr())
        self.singleton = _gde_object_get_instance_binding(self._gde_so, token, &self._binding_callbacks)


cdef class GodotMethodBindRet:
    cdef void *_owner
    cdef GDExtensionMethodBindPtr _gde_mb
    cdef StringName _method_name

    def __cinit__(self, GodotObject wrapper, str method_name, GDExtensionInt method_hash):
        self._owner = wrapper._owner
        self._method_name = stringname_from_str(method_name)
        self._gde_mb = _gde_classdb_get_method_bind(
            wrapper._class_name._native_ptr(), self._method_name._native_ptr(), method_hash)

    cpdef object _call_internal(self, tuple args):
        cdef Variant ret
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg
        
        with nogil:
            _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, &ret)
            _gde_mem_free(p_args)

        return pyobject_from_variant(ret)
    
    def __call__(self, *args):
        return self._call_internal(args)


cdef class GodotMethodBindNoRet(GodotMethodBindRet):
    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, NULL)
            _gde_mem_free(p_args)

cdef class GodotUtilityFunctionRet:
    cdef GDExtensionPtrUtilityFunction _gde_uf
    cdef StringName _function_name

    def __cinit__(self, str function_name, GDExtensionInt function_hash):
        self._function_name = stringname_from_str(function_name)
        self._gde_uf = _gde_variant_get_ptr_utility_function(self._function_name._native_ptr(), function_hash)
    
    cpdef object _call_internal(self, tuple args):
        cdef Variant ret
        cdef Variant arg
        cdef int i
        cdef int size = len(args)
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        for i in range(size):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            self._gde_uf(&ret, p_args, size)
            _gde_mem_free(p_args)

        return pyobject_from_variant(ret)

    def __call__(self, *args):
        return self._call_internal(args)


cdef class GodotUtilityFunctionNoRet(GodotUtilityFunctionRet):
    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef int i
        cdef int size = len(args)
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))
        
        for i in range(size):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            self._gde_uf(NULL, p_args, size)
            _gde_mem_free(p_args)

printraw = GodotUtilityFunctionNoRet('printraw', 2648703342)
print_rich = GodotUtilityFunctionNoRet('print_rich', 2648703342)
push_error = GodotUtilityFunctionNoRet('push_error', 2648703342)

include "sys_util.pxi"
