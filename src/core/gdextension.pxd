"""\
This module provides a reasonably low-level Python implementation
of the GDExtension API
"""

# TODO: Refactor to resemble GDExtension API structure closer

from libc.stdint cimport int8_t
from cpython cimport (
    PyObject, ref, pystate, PyLong_Check, PyLong_AsSsize_t,
    PyList_New, PyList_SET_ITEM, PyTuple_New, PyTuple_SET_ITEM
)
cimport cython
from cython.operator cimport dereference as deref

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
    """
    pass

cimport numpy

from binding cimport *
from godot_cpp cimport *
cimport godot_types as type_funcs


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


cdef enum ArgType:
    ARGTYPE_NIL
    ARGTYPE_BOOL
    ARGTYPE_INT
    ARGTYPE_FLOAT
    ARGTYPE_STRING
    ARGTYPE_VECTOR2
    ARGTYPE_VECTOR2I
    ARGTYPE_RECT2
    ARGTYPE_RECT2I
    ARGTYPE_VECTOR3
    ARGTYPE_VECTOR3I
    ARGTYPE_TRANSFORM2D
    ARGTYPE_VECTOR4
    ARGTYPE_VECTOR4I
    ARGTYPE_PLANE
    ARGTYPE_QUATERNION
    ARGTYPE_AABB
    ARGTYPE_BASIS
    ARGTYPE_TRANSFORM3D
    ARGTYPE_PROJECTION
    ARGTYPE_COLOR
    ARGTYPE_STRING_NAME
    ARGTYPE_NODE_PATH
    ARGTYPE_RID
    ARGTYPE_OBJECT
    ARGTYPE_CALLABLE
    ARGTYPE_SIGNAL
    ARGTYPE_DICTIONARY
    ARGTYPE_ARRAY
    ARGTYPE_PACKED_BYTE_ARRAY
    ARGTYPE_PACKED_INT32_ARRAY
    ARGTYPE_PACKED_INT64_ARRAY
    ARGTYPE_PACKED_FLOAT32_ARRAY
    ARGTYPE_PACKED_FLOAT64_ARRAY
    ARGTYPE_PACKED_STRING_ARRAY
    ARGTYPE_PACKED_VECTOR2_ARRAY
    ARGTYPE_PACKED_VECTOR3_ARRAY
    ARGTYPE_PACKED_COLOR_ARRAY
    ARGTYPE_PACKED_VECTOR4_ARRAY

    ARGTYPE_VARIANT

    ARGTYPE_MAX


cdef class Class:
    """
    Defines all Godot Engine's classes.

    NOTE: Although instances of `gdextension.Class` and its subclasses implement
    class functionality, they are still *objects* on the Python level.

    Only on the higher level (`godot` module) they would be wrapped as real Python
    classes.

    Works as a singleton, can't be instantiated directly: use `Class.get_class`
    in Cython or `Class._get_class` in Python to create/get instances
    `gdextension.Class`.

    Doesn't implement any GDExtension API calls by itself.

    Captures method, property (TODO) and signal (TODO) information,
    processes class inheritance chains.
    """
    cdef readonly dict __method_info__
    cdef readonly str __name__
    cdef readonly Class __inherits__

    cdef int initialize_class(self) except -1
    cpdef object get_method_info(self, method_name)

    @staticmethod
    cdef Class get_class(object name)


cdef public class Object [object GDPyObject, type GDPyObject_Type]:
    """
    Defines all Godot Engine's objects.

    Implements following GDExtension API calls:
        in `Object.__init__`:
            `global_get_singleton` (for singleton objects)
            `classdb_construct_object` (for all others)

    Captures method, property (TODO) and signal (TODO) information,
    processes class inheritance chains.
    """
    cdef void *_owner
    cdef void *_ref_owner  # According to gdextension_interface.h, if _owner is Ref, this would be real owner
    cdef bint is_singleton
    cdef readonly Class __godot_class__


cdef class EngineCallableBase:
    cdef readonly str __name__
    cdef readonly tuple type_info
    cdef int8_t[16] _type_info_opt


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
    """
    Defines all custom classes which extend the Godot Engine.
    Inherits `gdextendion.Class`

    Implements instance management callbacks in the ClassCreationInfo4 structure:
        `creation_info4.create_instance_func = &ExtensionClass.create_instance`
        `creation_info4.free_instance_func = &ExtensionClass.free_instance`
        `creation_info4.recreate_instance_func = &ExtensionClass.recreate_instance`

    Stores information about all new methods and class registration state.

    Implements all class registration calls and delegates them to `gdextension.ClassRegistrator`
    """
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


cdef class ExtensionClassRegistrator:
    """
    Registers `ExtensionClass` objects.

    Implements following GDExtension API calls:
        in `ExtensionClassRegistrator.__cinit__`:
            `classdb_register_extension_class4`
    """
    cdef str __name__
    cdef ExtensionClass registree
    cdef Class inherits

    cdef int register_method(self, func: types.FunctionType) except -1
    cdef int register_virtual_method(self, func: types.FunctionType) except -1


cdef public class Extension(Object) [object GDPyExtension, type GDPyExtension_Type]:
    """
    Defines all instances of `gdextension.Class`.

    Implements following GDExtension API calls:
        in `Extension.__init__`
            `classdb_construct_object` (of base class)
            `object_set_instance`
        in `Extension.__del__` and `Extension.destroy`
            `object_destroy`

    Implements virtual call callbacks in the ClassCreationInfo4 structure:
        `creation_info4.get_virtual_call_data_func = &Extension.get_virtual_call_data`
        `creation_info4.call_virtual_with_data_func = &Extension.call_virtual_with_data`
    """
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
    """"
    Defines all custom methods of `gdextension.Extension` objects.

    Implements following GDExtension API calls:
        in `ExtensionMethod.register`
            `classdb_register_extension_class_method`

    Implements `call`/`ptrcall` callbacks in the `ClassMethodInfo` structure.
    """
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
    cdef int8_t[16] _type_info_opt


cdef class BoundExtensionMethod(PythonCallableBase):
    cdef readonly Extension __self__

    cdef size_t get_argument_count(self) except -2


include "includes/engine_calls_header.pxi"
include "includes/python_calls_header.pxi"
