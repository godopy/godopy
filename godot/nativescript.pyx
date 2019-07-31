from godot_headers.gdnative_api cimport *

from .globals cimport (
    PyGodot, gdapi, nativescript_api as nsapi, nativescript_1_1_api as ns11api, _nativescript_handle as handle
)
from .cpp.core_types cimport String
from .core_types cimport _Wrapped, _PyWrapped, CythonTagDB, PythonTagDB, register_cython_type, register_python_type
from .bindings cimport _cython_bindings, _python_bindings

from cpython.object cimport PyObject, PyTypeObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.ref cimport Py_INCREF, Py_DECREF

from pygodot.utils cimport _init_dynamic_loading


cdef void *_instance_create(size_t type_tag, godot_object *instance, dict TagDB) except NULL:
    cdef type cls = TagDB[type_tag]
    cdef obj = cls()

    (<_Wrapped>obj)._owner = instance
    (<_Wrapped>obj)._type_tag = type_tag

    print('instance %s (%s) created: %s, %s' % (obj,
          hex(<size_t><void *>obj), hex(<size_t>instance), hex(<size_t>type_tag)))

    Py_INCREF(obj)
    return <void *>obj


# FIXME: Should be declared GDCALLINGCONV void *
cdef void *_cython_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, CythonTagDB)

cdef void *_python_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, PythonTagDB)

cdef void _wrapper_destroy(void *data, void *wrapper) nogil:
    if not wrapper:
        return

    with gil:
        Py_DECREF(<object>wrapper)

cdef void _wrapper_incref(void *data, void *wrapper) nogil:
    if not wrapper:
        return

    with gil:
        Py_INCREF(<object>wrapper)

cdef bool _wrapper_decref(void *data, void *wrapper) nogil:
    if not wrapper:
        return False

    with gil:
        Py_DECREF(<object>wrapper)

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


cpdef register_class(type cls):
    cdef void *method_data = <PyTypeObject *>cls
    cdef void *type_tag = method_data
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

    cls._register_methods()


cdef void *_cython_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, CythonTagDB)


cdef void *_python_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, PythonTagDB)


cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    if user_data:
        with gil: Py_DECREF(<object>user_data)


# Example of a public C function pointer declaration:
# cdef public:
#     ctypedef void (*cfunc_void_object_float)(object, float)


cdef test_method_call(type cls, object instance, fusedmethod method):
    if fusedmethod is Method__float:
        method(instance, 5)
