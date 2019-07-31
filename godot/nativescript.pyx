from godot_headers.gdnative_api cimport *

from .globals cimport PyGodot, gdapi, nativescript_api, nativescript_1_1_api, _nativescript_handle
from .cpp.core_types cimport String
from .core_types cimport _Wrapped
from .bindings cimport _cython_bindings, _python_bindings

from cpython.object cimport PyObject, PyTypeObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.ref cimport Py_INCREF, Py_DECREF

from pygodot.utils cimport _init_dynamic_loading


# FIXME: Should be declared GDCALLINGCONV void *
cdef void *_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        wrapper = _Wrapped()  # To skip __init__, call _Wrapped.__new__(_Wrapped)
        (<_Wrapped>wrapper)._owner = instance
        (<_Wrapped>wrapper)._type_tag = <void *>type_tag
        print('Godot wrapper %s created: inst 0x%x, #0x%x' % (wrapper, <size_t>instance, <size_t>type_tag))

    return <void *>wrapper

cdef void _wrapper_destroy(void *data, void *wrapper) nogil:
    if wrapper:
        with gil: Py_DECREF(<object>wrapper)

cdef void _wrapper_incref(void *data, void *wrapper) nogil:
    if wrapper:
        with gil: Py_INCREF(<object>wrapper)

cdef bool _wrapper_decref(void *data, void *wrapper) nogil:
    if wrapper:
        with gil: Py_DECREF(<object>wrapper)

    return False  # FIXME


cdef public cython_nativescript_init():
    cdef godot_instance_binding_functions binding_funcs = [
        &_wrapper_create,
        &_wrapper_destroy,
        &_wrapper_incref,
        &_wrapper_decref,
        NULL,  # void *data
        NULL   # void (*free_func)(void *)
    ]

    cdef int language_index = \
        nativescript_1_1_api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

    PyGodot.set_cython_language_index(language_index)

    _cython_bindings.__register_types()
    _cython_bindings.__init_method_bindings()


cdef public python_nativescript_init():
    cdef godot_instance_binding_functions binding_funcs = [
        &_wrapper_create,
        &_wrapper_destroy,
        &_wrapper_incref,
        &_wrapper_decref,
        NULL,  # void *data
        NULL   # void (*free_func)(void *)
    ]

    cdef int language_index = \
        nativescript_1_1_api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

    PyGodot.set_python_language_index(language_index)

    _python_bindings.__register_types()
    _python_bindings.__init_method_bindings()


cdef public generic_nativescript_init():
    _init_dynamic_loading()

    import gdlibrary

    if hasattr(gdlibrary, 'nativescript_init'):
        gdlibrary.nativescript_init()


cpdef register_class(type cls):
    cdef void *method_data = <PyTypeObject *>cls
    cdef void *type_tag = method_data

    cdef godot_instance_create_func create = [NULL, NULL, NULL]
    create.create_func = &_instance_func
    create.method_data = method_data

    cdef godot_instance_destroy_func destroy = [NULL, NULL, NULL]
    destroy.destroy_func = &_destroy_func
    destroy.method_data = method_data

    cdef bytes name = cls.__name__.encode('utf-8')
    cdef bytes base = cls.__bases__[0].__name__.encode('utf-8')

    nativescript_api.godot_nativescript_register_class(_nativescript_handle,
                                                       <const char *>name, <const char *>base, create, destroy)
    nativescript_1_1_api.godot_nativescript_set_type_tag(_nativescript_handle, <const char *>name, type_tag)

    cls._register_methods()


cdef void *_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        cls = <type>method_data
        obj = cls()
        (<_Wrapped>obj)._owner = instance
        (<_Wrapped>obj)._type_tag = <void *>cls

        Py_INCREF(obj)
        return <void *>obj


cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    if user_data:
        with gil: Py_DECREF(<object>user_data)


# Example of a public C function pointer declaration:
# cdef public:
#     ctypedef void (*cfunc_void_object_float)(object, float)


cdef test_method_call(type cls, object instance, fusedmethod method):
    if fusedmethod is Method__float:
        method(instance, 5)
