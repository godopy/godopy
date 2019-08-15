# cython: c_string_encoding=utf-8
from godot_headers.gdnative_api cimport *

from .core.globals cimport (
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
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem
from cpython.ref cimport Py_INCREF, Py_DECREF

from .utils cimport _init_dynamic_loading

from cython.operator cimport dereference as deref


cdef void *_instance_create(size_t type_tag, godot_object *instance, root_base, dict TagDB) except NULL:
    cdef type cls = TagDB[type_tag]
    cdef obj = cls.__new__(cls)  # Don't call __init__

    __instance_map[<size_t>instance] = obj

    (<_Wrapped>obj)._owner = instance
    (<_Wrapped>obj).___CLASS_IS_SCRIPT = True

    if hasattr(obj, '_init'):
        obj._init()

    print('instance created', obj)

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
    cdef size_t _owner;
    with gil:
        _owner = <size_t>(<_Wrapped>wrapper)._owner
        if _owner in __instance_map:
            del __instance_map[_owner]
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
        return _instance_create(<size_t>method_data, instance, _Wrapped, CythonTagDB)


cdef void *_python_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, _PyWrapped, PythonTagDB)


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
    cdef tuple __args = PyTuple_New(num_args)

    for i in range(num_args):
        # PyObject * casts are defined in `Variant::operator PyObject *() const` C++ method
        PyTuple_SetItem(__args, i, <object>args[i])

    return __args


cdef godot_variant _python_method_wrapper(godot_object *instance, void *method_data, void *self_data, int num_args, godot_variant **args) nogil:
    cdef Variant result

    with gil:
        self = <object>self_data
        method = <object>method_data

        # Variant casts from PyObject* are defined in Variant::Variant(const PyObject*) constructor
        result = <Variant>method(self, *__parse_args(num_args, args))

    return deref(<godot_variant *>&result)


cdef void _python_method_free(void *method_data) nogil:
    __decref_python_pointer(method_data)
