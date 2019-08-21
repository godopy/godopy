# cython: c_string_encoding=utf-8
from godot_headers.gdnative_api cimport *

from .globals cimport (
    Godot, PyGodot,
    gdapi, nativescript_api as nsapi, nativescript_1_1_api as ns11api,
    _nativescript_handle as handle
)
from .core.cpp_types cimport String, Variant
from .core._wrapped cimport _Wrapped, _PyWrapped
from .core.tag_db cimport CythonTagDB, PythonTagDB, __instance_map, register_cython_type, register_python_type
from .core.defs cimport VARIANT_OBJECT, VARIANT_NIL
from .core.signal_arguments cimport SignalArgument

from .bindings cimport _cython_bindings, _python_bindings

from libcpp cimport nullptr
from libc.string cimport memset
from cpython.object cimport PyObject, PyTypeObject
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.ref cimport Py_INCREF, Py_DECREF

from cython.operator cimport dereference as deref

cdef extern from *:
    """
#define GDCALLINGCONV_VOID_PTR GDCALLINGCONV void*
#define GDCALLINGCONV_VOID GDCALLINGCONV void
#define GDCALLINGCONV_BOOL GDCALLINGCONV bool
    """
    ctypedef void* GDCALLINGCONV_VOID_PTR
    ctypedef void GDCALLINGCONV_VOID
    ctypedef bint GDCALLINGCONV_BOOL


cdef void *_instance_create(size_t type_tag, godot_object *instance, dict TagDB, is_instance_binding=False) except NULL:
    cdef type cls = TagDB[type_tag]
    cdef obj = cls.__new__(cls)  # Don't call __init__

    __instance_map[<size_t>instance] = obj

    (<_Wrapped>obj)._owner = instance
    (<_Wrapped>obj).___CLASS_IS_SCRIPT = True

    # Check for _init in class dictionary, otherwise Object._init() will be called incorrectly
    if '_init' in obj.__class__.__dict__:
        obj._init()

    if is_instance_binding:
        print('instance binding CREATE', obj, hex(<size_t><void *>instance))

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

cdef GDCALLINGCONV_VOID_PTR _cython_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, CythonTagDB, is_instance_binding=True)

cdef GDCALLINGCONV_VOID_PTR _python_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, PythonTagDB, is_instance_binding=True)

cdef set __DESTROYED = set()
cdef GDCALLINGCONV_VOID _wrapper_destroy(void *data, void *wrapper) nogil:
    cdef size_t _owner;
    with gil:
        if <size_t>wrapper in __DESTROYED:
            # print("instance binding DESTROY recursive call", hex(<size_t>wrapper))
            __DESTROYED.remove(<size_t>wrapper)
            return

        print("instance binding DESTROY", hex(<size_t>wrapper), hex(<size_t>data), hex(<size_t>(<_Wrapped>wrapper)._owner))
        _owner = <size_t>(<_Wrapped>wrapper)._owner
        Py_DECREF(<object>wrapper)

        if _owner:
            (<_Wrapped>wrapper)._owner = NULL
            if _owner in __instance_map:
                del __instance_map[_owner]
            __DESTROYED.add(<size_t>wrapper)
            # FIXME: This will call _wrapper_destroy recursively, but this is the only way to clean up the _owner on the Godot side
            gdapi.godot_object_destroy(<godot_object *>_owner)


cdef GDCALLINGCONV_VOID _wrapper_incref(void *wrapper, void *owner) nogil:
    with gil:
       print("instance binding INCREF", hex(<size_t>wrapper), hex(<size_t>owner))
    __incref_python_pointer(wrapper)


cdef GDCALLINGCONV_BOOL _wrapper_decref(void *wrapper, void *owner) nogil:
    with gil:
        # FIXME: This is sometimes called before DESTROY without any previous calls to INCREF
        print("instance binding DECREF *IGNORED*", hex(<size_t>wrapper), hex(<size_t>owner))
    # __decref_python_pointer(wrapper)

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


cdef public cython_nativescript_terminate():
    pass


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


cdef public python_nativescript_terminate():
    pass


cdef public generic_nativescript_init():
    from importlib import import_module

    cdef _cython_bindings.ProjectSettings ps = _cython_bindings.ProjectSettings.get_singleton()
    gdlibrary_name = <object>ps.get_setting('python/config/gdnlib_module')
    gdlibrary = import_module(gdlibrary_name)

    if hasattr(gdlibrary, '_nativescript_init'):
        gdlibrary._nativescript_init()


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

    print('register class', name, base)

    # TODO: Set custom __init__ that woould call NativeScript.new and capture the _owner pointer from that

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
        return _instance_create(<size_t>method_data, instance, CythonTagDB)


cdef void *_python_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, PythonTagDB)


cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    cdef size_t _owner = <size_t>instance
    with gil:
        if _owner in __instance_map:
            del __instance_map[_owner]
    __decref_python_pointer(user_data)


cdef register_signal(type cls, str name, object args=()):
    cdef String _name = String(name)

    cdef godot_signal signal = [
        deref(<godot_string *>&_name),
        len(args),             # .num_args
        NULL,                  # .args
        0,                     # .num_default_args
        <godot_variant *>NULL  # .default_args
    ]

    if args:
        signal.args = <godot_signal_argument *>gdapi.godot_alloc(sizeof(godot_signal_argument) * signal.num_args)
        memset(<void *>signal.args, 0, sizeof(godot_signal_argument) * signal.num_args)

    for i, arg in enumerate(args):
        _set_signal_argument(&signal.args[i], arg)

    cdef bytes class_name = cls.__name__.encode('utf-8')
    nsapi.godot_nativescript_register_signal(handle, <const char *>class_name, &signal)

    for i, arg in enumerate(args):
        gdapi.godot_string_destroy(&signal.args[i].name)

        if arg.hint_string:
           gdapi.godot_string_destroy(&signal.args[i].hint_string)

    if args:
        gdapi.godot_free(signal.args)


def register_python_signal(type cls, str name, *args):
    register_signal(cls, name, args)


cdef inline _set_signal_argument(godot_signal_argument *sigarg, object _arg):
    cdef SignalArgument arg = _arg
    cdef String _name
    cdef String _hint_string
    cdef Variant _def_val

    _name = <String>arg.name
    gdapi.godot_string_new_copy(&sigarg.name, <godot_string *>&_name)

    print('set arg', arg.name, arg.type, VARIANT_OBJECT)
    sigarg.type = arg.type

    if arg.hint_string:
        _hint_string = String(arg.hint_string)
        gdapi.godot_string_new_copy(&sigarg.hint_string, <godot_string *>&_hint_string)

    # arg.usage = <godot_property_usage_flags>(<int>arg.usage | GODOT_PROPERTY_USAGE_SCRIPT_VARIABLE)

    sigarg.hint = arg.hint
    sigarg.usage = arg.usage

    cdef bint has_default_value = arg.default_value is not None

    if arg.type == VARIANT_OBJECT:
        # None is a valid default value for Objects
        has_default_value = arg.default_value != -1

    if has_default_value:
        _def_val = <Variant>arg.default_value
        sigarg.default_value = deref(<godot_variant *>&_def_val)


cdef register_property(type cls, const char *name, object default_value,
                       godot_method_rpc_mode rpc_mode=GODOT_METHOD_RPC_MODE_DISABLED,
                       godot_property_usage_flags usage=GODOT_PROPERTY_USAGE_DEFAULT,
                       godot_property_hint hint=GODOT_PROPERTY_HINT_NONE,
                       str hint_string=''):
    cdef String _hint_string = String(hint_string)
    cdef Variant def_val = <Variant>default_value

    usage = <godot_property_usage_flags>(<int>usage | GODOT_PROPERTY_USAGE_SCRIPT_VARIABLE)

    if def_val.get_type() == <Variant.Type>VARIANT_OBJECT:
        pass  # TODO: Set resource hints!

    cdef godot_property_attributes attr = [
        rpc_mode,
        def_val.get_type() if def_val.get_type() != <Variant.Type>VARIANT_NIL else VARIANT_OBJECT,
        hint,
        deref(<godot_string *>&_hint_string),
        usage,
        deref(<godot_variant *>&def_val)
    ]

    cdef str property_data = name
    Py_INCREF(property_data)

    cdef godot_property_set_func set_func = [NULL, NULL, NULL]
    set_func.method_data = <void *>property_data
    set_func.set_func = _property_setter
    set_func.free_func = _python_method_free

    cdef godot_property_get_func get_func = [NULL, NULL, NULL]
    get_func.method_data = <void *>property_data
    get_func.get_func = _property_getter

    cdef bytes class_name = cls.__name__.encode('utf-8')

    nsapi.godot_nativescript_register_property(handle, <const char *>class_name, name, &attr, set_func, get_func)


cdef godot_variant _property_getter(godot_object *object, void *method_data, void *self_data) nogil:
    cdef Variant result

    with gil:
        self = <object>self_data
        prop_name = <str>method_data
        # Variant casts from PyObject* are defined in `Variant::Variant(const PyObject*)` C++ constructor
        result = <Variant>getattr(self, prop_name)

    return deref(<godot_variant *>&result)


cdef void _property_setter(godot_object *object, void *method_data, void *self_data, godot_variant *value) nogil:
    cdef Variant _value = deref(<Variant *>value)
    with gil:
        self = <object>self_data
        prop_name = <str>method_data

        # PyObject * casts are defined in `Variant::operator PyObject *() const` C++ method
        setattr(self, prop_name, <object>_value)


def register_python_property(type cls, str name, object default_value,
                             godot_method_rpc_mode rpc_mode=GODOT_METHOD_RPC_MODE_DISABLED,
                             godot_property_usage_flags usage=GODOT_PROPERTY_USAGE_DEFAULT,
                             godot_property_hint hint=GODOT_PROPERTY_HINT_NONE,
                             str hint_string=''):
    register_property(cls, name, default_value, rpc_mode, usage, hint, hint_string)


ctypedef godot_variant (*__godot_wrapper_method)(godot_object *, void *, void *, int, godot_variant **) nogil

cdef _register_python_method(type cls, const char *name, object method, godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED):
    Py_INCREF(method)
    cdef godot_instance_method m = [<__godot_wrapper_method>_python_method_wrapper, <void *>method, _python_method_free]
    cdef godot_method_attributes attrs = [rpc_type]

    cdef bytes class_name = cls.__name__.encode('utf-8')
    nsapi.godot_nativescript_register_method(handle, <const char *>class_name, name, attrs, m)


def register_python_method(type cls, str method_name, *, object method=None, godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED):
    if method is None:
        method = getattr(cls, method_name)
    cdef bytes _method_name = method_name.encode('utf-8')
    return _register_python_method(cls, <const char *>_method_name, method, rpc_type)


cdef tuple __parse_args(int num_args, godot_variant **args):
    cdef Py_ssize_t i
    cdef object __args = PyTuple_New(num_args)
    Py_INCREF(__args)

    for i in range(num_args):
        # PyObject * casts are defined in `Variant::operator PyObject *() const` C++ method
        PyTuple_SET_ITEM(__args, i, <object>deref(<Variant *>args[i]))

    return __args


cdef Variant _python_method_wrapper(godot_object *instance, void *method_data, void *self_data, int num_args, godot_variant **args) nogil:
    with gil:
        self = <object>self_data
        method = <object>method_data
        Py_INCREF(self)  # Make sure there are no leaks here

        # Variant casts from PyObject* are defined in Variant::Variant(const PyObject*) constructor
        return <Variant>method(self, *__parse_args(num_args, args))


cdef void _python_method_free(void *method_data) nogil:
    __decref_python_pointer(method_data)
