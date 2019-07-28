from godot_headers.gdnative_api cimport *
from .globals cimport gdapi, nativescript_api, nativescript_1_1_api, _nativescript_handle as handle
from ._core cimport _Wrapped
from .bindings.cython cimport nodes

# from pygodot.utils import _pyprint as print

from cpython.object cimport PyObject, PyTypeObject

__keepalive = set()


cpdef object register_method(object cls, object method, godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED):
    _register_method(cls, method, rpc_type)


cpdef object register_class(object cls):
    _register_class(cls)


cdef inline cls2typetag(cls):
    cdef PyObject *_type_tag = <PyObject *>cls
    return <size_t>_type_tag

cdef inline __keep_ptr(object obj):
    # TODO: Py_INCREF
    print('[PyNS] Create PyObject', obj)
    __keepalive.add(obj)

cdef inline __free_ptr(void *ptr):
    # TODO: Py_DECREF
    print('[PyNS] Free PyObject', <object>ptr)
    if <object>ptr in __keepalive:
        __keepalive.remove(<object>ptr)

cdef inline set_wrapper_tags(PyObject *o, godot_object *_owner, size_t _type_tag):
    cdef _Wrapped wrapper = <_Wrapped>o
    wrapper._owner = _owner
    wrapper._type_tag = _type_tag

### Initializer (not used, for future reference)
cdef class PyGodotGlobal(nodes.Node):
    cdef void _ready(self):
        print("GLOBAL READY!")

    @classmethod
    def _register_methods(cls):
        cdef godot_instance_method method = [PyGodotGlobal_ready_wrapper, NULL, NULL]
        cdef godot_method_attributes attrs = [GODOT_METHOD_RPC_MODE_DISABLED]

        nativescript_api.godot_nativescript_register_method(handle, "PyGodotGlobal", "_ready", attrs, method)

cdef godot_variant PyGodotGlobal_ready_wrapper(godot_object *o, void *md, void *p_instance, int n,
                                               godot_variant **args) nogil:
    with gil:
        instance = <PyGodotGlobal>p_instance
        instance._ready()

    cdef godot_variant ret
    gdapi.godot_variant_new_nil(&ret)
    return ret

cdef public object _clsdef_PyGodotGlobal = PyGodotGlobal

### Class Registration

cdef public int _register_class(object cls) except -1:
    # Add checks!
    cdef void *method_data = <PyObject *>cls
    cdef void *type_tag = method_data

    cdef godot_instance_create_func create = [NULL, NULL, NULL]
    create.create_func = &_instance_func
    create.method_data = method_data

    cdef godot_instance_destroy_func destroy = [NULL, NULL, NULL]
    destroy.destroy_func = &_destroy_func
    destroy.method_data = method_data

    cdef bytes name = cls.__name__.encode('utf-8')
    cdef bytes base = cls.__bases__[0].__name__.encode('utf-8')

    nativescript_api.godot_nativescript_register_class(handle, <const char *>name, <const char *>base, create, destroy)
    nativescript_1_1_api.godot_nativescript_set_type_tag(handle, <const char *>name, type_tag)

    cls._register_methods()
    return 0

cdef void *_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        cls = <object>method_data
        obj = cls()

        set_wrapper_tags(<PyObject *>obj, instance, cls2typetag(cls))

        __keep_ptr(obj)
        return <PyObject *>obj

cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    with gil:
        __free_ptr(user_data)

cdef public:
    ctypedef void (*cfunc_void_object_float)(object, float)


### Method registration

cdef public object _register_method(object cls, object method, godot_method_rpc_mode rpc_type):
    cdef str uname = method.__name__
    __keep_ptr(method)

    cdef godot_instance_method m = [NULL, NULL, NULL]

    m.method = _method_wrapper
    m.method_data = <void *><PyObject *>method
    m.free_func = &_method_destroy

    cdef godot_method_attributes attrs = [rpc_type]
    cdef bytes name = uname.encode('utf-8')
    cdef bytes classname = cls.__name__.encode('utf-8')

    nativescript_api.godot_nativescript_register_method(handle, <const char *>classname, <const char *>name, attrs, m)


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

        return convert_result(method.__call__(python_instance, *parse_args(num_args, args)))
