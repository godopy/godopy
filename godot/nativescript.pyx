from godot_headers.gdnative_api cimport *

from .globals cimport (
    PyGodot, gdapi, nativescript_api as nsapi, nativescript_1_1_api as ns11api, _nativescript_handle as handle
)
from .cpp.core_types cimport String, Variant as CVariant
from .core_types cimport _Wrapped, _PyWrapped, CythonTagDB, PythonTagDB, register_cython_type, register_python_type
from .bindings cimport _cython_bindings, _python_bindings

from libcpp cimport nullptr
from cpython.object cimport PyObject, PyTypeObject
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem
from cpython.ref cimport Py_INCREF, Py_DECREF

from pygodot.utils cimport _init_dynamic_loading


cdef void *_instance_create(size_t type_tag, godot_object *instance, root_base, dict TagDB) except NULL:
    cdef type cls = TagDB[type_tag]
    cdef obj = cls.__new__(cls)  # Don't call __init__

    if root_base is _PyWrapped:
        (<_PyWrapped>obj)._owner = instance
    else:
        (<_Wrapped>obj)._owner = instance

    print('instance %s (%s) created: %s, %s' % (obj,
          hex(<size_t><void *>obj), hex(<size_t>instance), hex(<size_t>type_tag)))

    # Cython manages refs automatically and decrements created objects on function exit,
    # therefore INCREF is required to keep the object alive.
    # An alternative would be to instatiate via a direct C API call, eg PyObject_New or PyObject_Call
    Py_INCREF(obj)
    return <void *>obj


cdef inline void __decref_python_pointer(void *ptr) nogil:
    if not ptr: return
    with gil: Py_DECREF(<object>ptr)

cdef inline void __incref_python_pointer(void *ptr) nogil:
    if not ptr: return
    with gil: Py_INCREF(<object>ptr)

# FIXME: Should be declared GDCALLINGCONV void *
cdef void *_cython_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, _Wrapped, CythonTagDB)

cdef void *_python_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, _PyWrapped, PythonTagDB)

cdef void _wrapper_destroy(void *data, void *wrapper) nogil:
    __decref_python_pointer(wrapper)

cdef void _wrapper_incref(void *data, void *wrapper) nogil:
    __incref_python_pointer(wrapper)

cdef bool _wrapper_decref(void *data, void *wrapper) nogil:
    __decref_python_pointer(wrapper)
    return False  # FIXME


cdef public cython_nativescript_init():
    cdef godot_instance_binding_functions binding_funcs = [
        &_cython_wrapper_create,
        &_wrapper_destroy,
        &_wrapper_incref,
        &_wrapper_decref,
        NULL,  # void *data
        NULL   # void (*free_func)(void *)
    ]

    cdef int language_index = ns11api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

    PyGodot.set_cython_language_index(language_index)

    _cython_bindings.__register_types()
    _cython_bindings.__init_method_bindings()


cdef public python_nativescript_init():
    cdef godot_instance_binding_functions binding_funcs = [
        &_python_wrapper_create,
        &_wrapper_destroy,
        &_wrapper_incref,
        &_wrapper_decref,
        NULL,  # void *data
        NULL   # void (*free_func)(void *)
    ]

    cdef int language_index = ns11api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

    PyGodot.set_python_language_index(language_index)

    _python_bindings.__register_types()
    _python_bindings.__init_method_bindings()


cdef public generic_nativescript_init():
    _init_dynamic_loading()

    import gdlibrary

    if hasattr(gdlibrary, 'nativescript_init'):
        gdlibrary.nativescript_init()


cdef __default_registration_function(type cls):
    cls._register_methods()


cdef _register_class(type cls, methods_registration_function registration_func):
    cdef void *method_data = <void *>cls
    cdef bint is_python = issubclass(cls, _PyWrapped)

    cdef godot_instance_create_func create = [NULL, NULL, NULL]

    if is_python:
        create.create_func = &_python_instance_func
    else:
        create.create_func = &_cython_instance_func
    create.method_data = method_data

    cdef godot_instance_destroy_func destroy = [NULL, NULL, NULL]
    destroy.destroy_func = &_destroy_func
    destroy.method_data = method_data

    cdef bytes name = cls.__name__.encode('utf-8')
    cdef bytes base = cls.__bases__[0].__name__.encode('utf-8')

    nsapi.godot_nativescript_register_class(handle, <const char *>name, <const char *>base, create, destroy)

    if is_python:
        register_python_type(cls)
    else:
        register_cython_type(cls)

    if methods_registration_function is _regfunc_classobj:
        registration_func(cls)
    elif methods_registration_function is _regfunc_static:
        registration_func()


cpdef register_class(type cls):
    _register_class(cls, __default_registration_function)


cdef void *_cython_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, _Wrapped, CythonTagDB)


cdef void *_python_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, _PyWrapped, PythonTagDB)


cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    __decref_python_pointer(user_data)


# Example of a public C function pointer declaration:
# cdef public:
#     ctypedef void (*cfunc_void_object_float)(object, float)


cdef test_method_call(type cls, object instance, fusedmethod method):
    if fusedmethod is Method__float:
        method(instance, 5)

# ctypedef godot_variant (*__godot_wrapper_method)(godot_object *, void *, void *, int, godot_variant **) nogil;

cdef _register_python_method(type cls, const char *name, object method, godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED):
    Py_INCREF(method)
    cdef godot_instance_method m = [_python_method_wrapper, <void *>method, _python_method_free]
    cdef godot_method_attributes attrs = [rpc_type]

    cdef bytes class_name = cls.__name__.encode('utf-8')
    nsapi.godot_nativescript_register_method(handle, <const char *>class_name, name, attrs, m)


# cdef extern from "Godot.hpp" namespace "godot":
#     cdef cppclass _ArgCast[T]:
#         @staticmethod
#         T _arg_cast(CVariant a)


cdef tuple __parse_args(int num_args, godot_variant **args):
    cdef Py_ssize_t i
    cdef godot_variant_type t
    cdef tuple __args = PyTuple_New(num_args)
    cdef object arg

    for i in range(num_args):
        arg = None
        t = gdapi.godot_variant_get_type(args[i])

        if t == GODOT_VARIANT_TYPE_REAL:
            arg = <float>gdapi.godot_variant_as_real(args[i])
        elif t == GODOT_VARIANT_TYPE_INT:
            arg = <int>gdapi.godot_variant_as_int(args[i])
        elif t == GODOT_VARIANT_TYPE_BOOL:
            arg = <bool>gdapi.godot_variant_as_bool(args[i])

        Py_INCREF(arg)
        PyTuple_SetItem(__args, i, arg)

    return __args


cdef godot_variant _python_method_wrapper(godot_object *instance, void *method_data, void *self_data, int num_args, godot_variant **args) nogil:
    cdef CVariant result

    with gil:
        self = <object>self_data
        method = <object>method_data

        # Variant casts from PyObject* are defined in Variant::Variant(const PyObject*) constructor
        result = <CVariant>method(self, *__parse_args(num_args, args))

    cdef godot_variant *ret = <godot_variant *>&result
    return ret[0]


cdef void _python_method_free(void *method_data) nogil:
    __decref_python_pointer(method_data)
