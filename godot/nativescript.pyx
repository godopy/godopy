from godot_headers.gdnative_api cimport *

from .globals cimport (
    Godot, gdapi, nativescript_api, nativescript_1_1_api,
    _nativescript_handle,
    _cython_language_index, _python_language_index
)
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
        (<_Wrapped>wrapper)._type_tag = <size_t>type_tag
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

    _cython_language_index = \
        nativescript_1_1_api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

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

    _python_language_index = \
        nativescript_1_1_api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

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


cdef inline cls2typetag(cls):
    cdef PyObject *_type_tag = <PyObject *>cls
    return <size_t>_type_tag


cdef inline __keep_ptr(object obj):
    print('[ns-refs] incref', obj)
    Py_INCREF(obj)


cdef inline __free_ptr(void *ptr):
    print('[ns-refs] decref', <object>ptr)
    Py_DECREF(<object>ptr)


cdef inline set_wrapper_tags(PyObject *o, godot_object *_owner, size_t _type_tag):
    cdef _Wrapped wrapper = <_Wrapped>o
    wrapper._owner = _owner
    wrapper._type_tag = _type_tag


### Initializer (not used, for future reference)
# cdef class PyGodotGlobal(nodes.Node):
#     cdef _ready(self):
#         print("GLOBAL READY!")

#     @classmethod
#     def _register_methods(cls):
#         cdef godot_instance_method method = [PyGodotGlobal_ready_wrapper, NULL, NULL]
#         cdef godot_method_attributes attrs = [GODOT_METHOD_RPC_MODE_DISABLED]

#         nativescript_api.godot_nativescript_register_method(handle, "PyGodotGlobal", "_ready", attrs, method)

# cdef godot_variant PyGodotGlobal_ready_wrapper(godot_object *o, void *md, void *p_instance, int n,
#                                                godot_variant **args) nogil:
#     with gil:
#         instance = <PyGodotGlobal>p_instance
#         instance._ready()

#     cdef godot_variant ret
#     gdapi.godot_variant_new_nil(&ret)
#     return ret

# cdef public object _clsdef_PyGodotGlobal = PyGodotGlobal

### Class Registration


cdef void *_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        cls = <type>method_data
        obj = cls()

        set_wrapper_tags(<PyObject *>obj, instance, cls2typetag(cls))

        __keep_ptr(obj)
        return <void *>obj


cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    with gil:
        __free_ptr(user_data)


### Method registration

# Example of a public C function pointer declaration:
# cdef public:
#     ctypedef void (*cfunc_void_object_float)(object, float)


cdef test_method_call(type cls, object instance, object method):
    method(instance, 5)


# Important note: passing cdefs as objects forces them to be wrapped in Cython-generated defs!
cpdef register_method(type cls, str name, object method=None,
                      godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED):
    if method is None:
        method = getattr(cls, method)

    __keep_ptr(method)

    cdef godot_instance_method m = [NULL, NULL, NULL]

    m.method = _method_wrapper
    m.method_data = <void *>method
    m.free_func = &_method_destroy

    cdef godot_method_attributes attrs = [rpc_type]
    cdef bytes _name = name.encode('utf-8')
    cdef bytes class_name = cls.__name__.encode('utf-8')

    nativescript_api.godot_nativescript_register_method(_nativescript_handle, <const char *>class_name,
                                                        <const char *>_name, attrs, m)


cdef void _method_destroy(void *method_data) nogil:
    with gil:
        __free_ptr(method_data)


cdef list parse_args(int num_args, godot_variant **args):
    cdef godot_variant_type t;
    pyargs = []

    cdef int i
    for i in range(num_args):
        t = gdapi.godot_variant_get_type(args[i])

        # TODO: all other possible conversions
        if t == GODOT_VARIANT_TYPE_REAL:
            pyargs.append(<float>gdapi.godot_variant_as_real(args[i]))
        else:
            pyargs.append(None)

    return pyargs


cdef godot_variant convert_result(object result):
    cdef godot_variant gd_result

    if False:
        pass # TODO
    else:
        gdapi.godot_variant_new_nil(&gd_result)

    return gd_result


cdef godot_variant _method_wrapper(godot_object *instance, void *method_data, void *user_data,
                                   int num_args, godot_variant **args) nogil:
    with gil:
        python_instance = <object>user_data
        method = <object>method_data

        return convert_result(method(python_instance, *parse_args(num_args, args)))
