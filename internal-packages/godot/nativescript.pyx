# cython: c_string_encoding=utf-8
from godot_headers.gdnative_api cimport *

from .globals cimport (
    Godot, PyGodot, WARN_PRINT,
    gdapi, gdnlib, nativescript_api as nsapi, nativescript_1_1_api as ns11api,
    _cython_language_index as CYTHON_IDX, _python_language_index as PYTHON_IDX,
    _nativescript_handle as handle
)
from .core.cpp_types cimport String, Variant
from .core._wrapped cimport _Wrapped, _PyWrapped
from .core.tag_db cimport (
    register_cython_type, register_python_type, get_cython_type, get_python_type,
    register_godot_instance, unregister_godot_instance, is_godot_instance_registered, replace_python_instance,
    clear_cython, clear_python, clear_instance_map
)
from .core.defs cimport VARIANT_OBJECT, VARIANT_NIL
from .core.signals cimport SignalArgument

from .core._meta cimport __tp_dict, PyType_Modified
from .core._debug cimport __ob_refcnt

from .bindings cimport _cython_bindings, _python_bindings

from libcpp cimport nullptr
from libc.string cimport memset
from cpython.object cimport PyObject, PyTypeObject
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.ref cimport Py_INCREF, Py_DECREF, Py_CLEAR, Py_XINCREF, Py_XDECREF

from cython.operator cimport dereference as deref

DEF INSTANCE_BINDING_REFCNT_HOOKS = False

cdef extern from *:
    """
#define GDCALLINGCONV_VOID_PTR GDCALLINGCONV void*
#define GDCALLINGCONV_VOID GDCALLINGCONV void
#define GDCALLINGCONV_BOOL GDCALLINGCONV bool
    """
    ctypedef void* GDCALLINGCONV_VOID_PTR
    ctypedef void GDCALLINGCONV_VOID
    ctypedef bint GDCALLINGCONV_BOOL


cdef _cython_bindings.GDNativeLibrary cython_gdnlib = None
cdef _python_bindings.GDNativeLibrary python_gdnlib = None

# cdef set __CYTHON_REFS = set()
# cdef set __PYTHON_REFS = set()

cdef void *_instance_create(size_t type_tag, godot_object *instance, language_index, is_instance_binding=False) except NULL:
    cdef bint is_python = language_index == PYTHON_IDX
    cdef type cls = get_python_type(type_tag) if is_python else get_cython_type(type_tag)

    cdef bint is_python_reference = issubclass(cls, _python_bindings.Reference)
    cdef bint is_cython_reference = issubclass(cls, _cython_bindings.Reference)

    cdef _Wrapped obj = <_Wrapped>cls.__new__(cls)

    register_godot_instance(instance, obj)

    obj._owner = instance
    obj._owner_allocated = False
    obj.___CLASS_IS_SCRIPT = not is_instance_binding

    # if is_python_reference and not <void *>obj._owner == gdnlib:
    #     __PYTHON_REFS.add(<size_t>obj._owner)
    # elif is_cython_reference and not <void *>obj._owner == gdnlib:
    #     __CYTHON_REFS.add(<size_t>obj._owner)

    if is_instance_binding:
        if <void *>obj._owner != gdnlib and (is_python_reference or is_cython_reference):
            obj._owner_allocated = True
    #     print('instance binding CREATE', obj, hex(<size_t>obj._owner), cls, __ob_refcnt(obj), obj._owner_allocated)
    # else:
    #     print('instance CREATE', obj, hex(<size_t>instance), cls, __ob_refcnt(obj))

    # Check for _init in class dictionary, otherwise Object._init() will be called incorrectly
    if '_init' in cls.__dict__:
        obj._init()

    # Cython manages refs automatically and decrements created objects on function exit,
    # therefore INCREF is required to keep the object alive.
    Py_INCREF(obj)
    return <void *>obj


cdef GDCALLINGCONV_VOID_PTR _cython_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, CYTHON_IDX, is_instance_binding=True)


cdef GDCALLINGCONV_VOID_PTR _python_wrapper_create(void *data, const void *type_tag, godot_object *instance) nogil:
    with gil:
        return _instance_create(<size_t>type_tag, instance, PYTHON_IDX, is_instance_binding=True)


cdef GDCALLINGCONV_VOID _wrapper_destroy(void *data, void *wrapper) nogil:
    cdef size_t _owner;
    cdef PyObject *p_wrapper = <PyObject *>wrapper
    cdef int expected_refcnt = 1

    with gil:
        _owner = <size_t>(<_Wrapped>wrapper)._owner

        if _owner == 0:
            # print("instance binding DESTROY no owner", <object>wrapper, __ob_refcnt(<object>wrapper))
            if __ob_refcnt(<object>wrapper) > expected_refcnt:
                WARN_PRINT("Possible memory leak: reference count for %r is too large: expected %d, got %d" %
                           (<object>wrapper, expected_refcnt, __ob_refcnt(<object>wrapper)))
            elif __ob_refcnt(<object>wrapper) < expected_refcnt:
                WARN_PRINT("Reference count for %r is too small: expected %d, got %d" %
                           (<object>wrapper, expected_refcnt, __ob_refcnt(<object>wrapper)))
            Py_CLEAR(p_wrapper)
            return

        # print(
        #     "instance binding DESTROY", <object>wrapper, __ob_refcnt(<object>wrapper),
        #     hex(_owner), is_godot_instance_registered(_owner)
        # )

        if is_godot_instance_registered(_owner):
            expected_refcnt += 1

        if __ob_refcnt(<object>wrapper) > expected_refcnt:
            WARN_PRINT("Possible memory leak: reference count for %r is too large: expected %d, got %d" %
                       (<object>wrapper, expected_refcnt, __ob_refcnt(<object>wrapper)))
        elif __ob_refcnt(<object>wrapper) < expected_refcnt:
            WARN_PRINT("Reference count for %r is too small: expected at least %d, got %d" %
                       (<object>wrapper, expected_refcnt, __ob_refcnt(<object>wrapper)))

        unregister_godot_instance(<godot_object *>_owner)
        Py_CLEAR(p_wrapper)

        (<_Wrapped>wrapper)._owner = NULL


cdef void _destroy_func(godot_object *instance, void *method_data, void *user_data) nogil:
    cdef size_t _owner = <size_t>instance
    cdef PyObject *p_wrapper = <PyObject *>user_data

    with gil:
        # print('instance DESTROY', <object>user_data, __ob_refcnt(<object>user_data), is_godot_instance_registered(_owner))
        (<_Wrapped>user_data)._owner = NULL

        unregister_godot_instance(<godot_object *>_owner)
        Py_CLEAR(p_wrapper)


cdef void _python_method_free(void *method_data) nogil:
    cdef PyObject *p_method = <PyObject *>method_data

    with gil:
        Py_CLEAR(p_method)


IF INSTANCE_BINDING_REFCNT_HOOKS:
    cdef GDCALLINGCONV_VOID _wrapper_incref(void *wrapper, void *owner) nogil:
        cdef PyObject *p_wrapper = <PyObject *>wrapper
        cdef size_t _owner = <size_t>owner

        with gil:
            Py_XINCREF(p_wrapper)


    cdef GDCALLINGCONV_BOOL _wrapper_decref(void *wrapper, void *owner) nogil:
        cdef PyObject *p_wrapper = <PyObject *>wrapper
        cdef size_t _owner = <size_t>owner
        cdef int min_refcount = 2
        cdef bool ret

        with gil:
            if p_wrapper == NULL:
                WARN_PRINT('NULL pointer in DECREF')
                return False

            if is_godot_instance_registered(<size_t>owner):
                min_refcount += 1

            if __ob_refcnt(<object>wrapper) < min_refcount:
                WARN_PRINT("Reference count for %r is too small to decrement: expected at least %d, got %d" %
                           (<object>wrapper, min_refcount, __ob_refcnt(<object>wrapper)))
            else:
                Py_XDECREF(p_wrapper)

            # should_destroy = False

            # if _owner in __PYTHON_REFS:
            #     print('UNREF PYTHON', <object>wrapper, hex(_owner))
            #     should_destroy = (<_python_bindings.Reference>wrapper).unreference()
            #     print('UNREFED', should_destroy)
            # elif _owner in __CYTHON_REFS:
            #     print('UNREF CYTHON', <object>wrapper, hex(_owner))
            #     should_destroy = (<_cython_bindings.Reference>wrapper).unreference()
            #     print('UNREFED', should_destroy)

            # if should_destroy:
            #     unregister_godot_instance(<godot_object *>_owner)
            #     Py_CLEAR(p_wrapper)
            #     return True

            ret = (__ob_refcnt(<object>wrapper) <= min_refcount)
            return ret


cdef public cython_nativescript_init():
    global cython_gdnlib

    IF INSTANCE_BINDING_REFCNT_HOOKS:
        cdef godot_instance_binding_functions binding_funcs = [
            &_cython_wrapper_create,
            &_wrapper_destroy,
            &_wrapper_incref,
            &_wrapper_decref,
            NULL,  # void *data
            NULL   # void (*free_func)(void *)
        ]
    ELSE:
        cdef godot_instance_binding_functions binding_funcs = [
            &_cython_wrapper_create,
            &_wrapper_destroy,
            NULL,  # &_wrapper_incref,
            NULL,  # &_wrapper_decref,
            NULL,  # void *data
            NULL   # void (*free_func)(void *)
        ]


    cdef int language_index = ns11api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

    PyGodot.set_cython_language_index(language_index)

    _cython_bindings.__register_types()
    _cython_bindings.__init_method_bindings()

    cython_gdnlib = <_cython_bindings.GDNativeLibrary>ns11api.godot_nativescript_get_instance_binding_data(
        CYTHON_IDX,
        <godot_object *>gdnlib
    )


cdef public cython_nativescript_terminate():
    global cython_gdnlib

    clear_cython()
    clear_instance_map()

    cython_gdnlib = None


cdef public python_nativescript_init():
    global python_gdnlib

    IF INSTANCE_BINDING_REFCNT_HOOKS:
        cdef godot_instance_binding_functions binding_funcs = [
            &_python_wrapper_create,
            &_wrapper_destroy,
            &_wrapper_incref,
            &_wrapper_decref,
            NULL,  # void *data
            NULL   # void (*free_func)(void *)
        ]
    ELSE:
        cdef godot_instance_binding_functions binding_funcs = [
            &_python_wrapper_create,
            &_wrapper_destroy,
            NULL,  # &_wrapper_incref,
            NULL,  # &_wrapper_decref,
            NULL,  # void *data
            NULL   # void (*free_func)(void *)
        ]

    cdef int language_index = ns11api.godot_nativescript_register_instance_binding_data_functions(binding_funcs)

    PyGodot.set_python_language_index(language_index)

    _python_bindings.__register_types()
    _python_bindings.__init_method_bindings()

    python_gdnlib = <_python_bindings.GDNativeLibrary>ns11api.godot_nativescript_get_instance_binding_data(
        PYTHON_IDX,
        <godot_object *>gdnlib
    )


cdef public python_nativescript_terminate():
    import gc

    global python_gdnlib

    # Some Python objects may depend on Godot objects (eg PoolArray accessors)
    # and they must be collected before NativeScript is terminated,
    # otherwise Python would try to collect them during PyFinalizeEx which will cause a SIGSEGV on exit
    gc.collect()

    clear_python()

    python_gdnlib = None


cdef public generic_nativescript_init():
    from importlib import import_module

    assert _cython_bindings.ProjectSettings is not None

    gdlibrary_name = <object>_cython_bindings.ProjectSettings.get_setting('python/config/gdnlib_module')
    gdlibrary = import_module(gdlibrary_name)

    if hasattr(gdlibrary, '_nativescript_init'):
        gdlibrary._nativescript_init()


cdef object _generate_python_get_script_func(cls):
    def _get_script() -> _python_bindings.NativeScript:
        cdef _python_bindings.NativeScript script = _python_bindings.NativeScript(own_memory=False)
        cdef size_t script_owner = <size_t>script._owner

        assert python_gdnlib is not None, python_gdnlib

        script.set_library(python_gdnlib)
        script.set_class_name(cls.__name__)

        return script

    return _get_script


cdef object _generate_cython_get_script_func(cls):
    def _get_script() -> _cython_bindings.NativeScript:
        cdef _cython_bindings.NativeScript script = _cython_bindings.NativeScript(own_memory=False)
        cdef size_t script_owner = <size_t>script._owner

        assert cython_gdnlib is not None, cython_gdnlib

        script.set_library(cython_gdnlib)
        script.set_class_name(cls.__name__)

        return script

    return _get_script


cdef object _generate_python_init_func(cls):
    cdef size_t owner

    def __init__(self, own_memory=False):
        cdef _python_bindings.NativeScript script = cls._get_script()

        # print('SCRIPT', hex(script_owner), __ob_refcnt(script))

        cdef godot_object *owner = script._new_instance()
        (<_Wrapped>self)._owner = owner;
        (<_Wrapped>self)._owner_allocated = own_memory

        # Clean Python object created in script._new_instance()
        replace_python_instance(owner, self)

        # print(self, 'OWNER', hex(<size_t>owner), __ob_refcnt(script), self, __ob_refcnt(self))

    return __init__



cdef object _generate_cython_preinit_func(cls):
    cdef size_t owner

    def _preinit(self, own_memory=False):
        cdef _cython_bindings.NativeScript script = cls._get_script()

        cdef godot_object *owner = script._new_instance()
        (<_Wrapped>self)._owner = owner;
        (<_Wrapped>self)._owner_allocated = own_memory

        # Clean Python object created in script._new_instance()
        replace_python_instance(owner, self)

        # print(self, 'OWNER', hex(<size_t>owner), __ob_refcnt(script), self, __ob_refcnt(self))

    return _preinit


cdef _register_class(type cls, tool_class=False):
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

    if len(cls.__bases__) > 1:
        raise RuntimeError("Multiple inheritance is not allowed in Godot binding subclasses")
    elif not cls.__bases__:
        raise RuntimeError("Godot binding base class is required")

    cdef bytes name = cls.__name__.encode('utf-8')
    cdef bytes base = cls.__bases__[0].__name__.encode('utf-8')

    # print('register class', name, base)

    init_func_set = False

    if is_python and '__init__' in cls.__dict__:
        WARN_PRINT("overwriting %r, please use '_init' instead of '__init__'" % cls.__init__)
        cls._init = __init__
    elif cls.__init__.__qualname__.startswith(cls.__name__):
        raise RuntimeError("%r is not allowed in Godot binding subclasses, did you mean %r" %
                           (cls.__init__.__qualname__, cls.__init__.__qualname__.replace('__init__', '_init')))

    if is_python:
        cls._get_script = staticmethod(_generate_python_get_script_func(cls))
        cls.__init__ = _generate_python_init_func(cls)
    else:
        __tp_dict(cls)['_get_script'] = staticmethod(_generate_cython_get_script_func(cls))
        __tp_dict(cls)['_preinit'] = _generate_cython_preinit_func(cls)
        PyType_Modified(cls)

    if tool_class:
        nsapi.godot_nativescript_register_tool_class(handle, <const char *>name, <const char *>base, create, destroy)
    else:
        nsapi.godot_nativescript_register_class(handle, <const char *>name, <const char *>base, create, destroy)

    if is_python:
        register_python_type(cls)
    else:
        register_cython_type(cls)

    cls._register_methods()


cpdef register_class(type cls):
    _register_class(cls, tool_class=False)


cpdef register_tool_class(type cls):
    _register_class(cls, tool_class=True)


cdef void *_cython_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, CYTHON_IDX)


cdef void *_python_instance_func(godot_object *instance, void *method_data) nogil:
    with gil:
        return _instance_create(<size_t>method_data, instance, PYTHON_IDX)


cdef public object _register_python_signal(type cls, str name, tuple args):
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
    # print('register_signal', class_name, name)
    nsapi.godot_nativescript_register_signal(handle, <const char *>class_name, &signal)

    for i, arg in enumerate(args):
        gdapi.godot_string_destroy(&signal.args[i].name)

        if arg.hint_string:
           gdapi.godot_string_destroy(&signal.args[i].hint_string)

    if args:
        gdapi.godot_free(signal.args)


def register_signal(type cls, str name, *args):
    _register_python_signal(cls, name, args)


cdef inline _set_signal_argument(godot_signal_argument *sigarg, object _arg):
    cdef SignalArgument arg = _arg
    cdef String _name
    cdef String _hint_string
    cdef Variant _def_val

    _name = <String>arg.name
    gdapi.godot_string_new_copy(&sigarg.name, <godot_string *>&_name)

    # print('set arg', arg.name, arg.type, VARIANT_OBJECT)
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


cdef public object _register_python_property(type cls, const char *name, object default_value, godot_method_rpc_mode rpc_mode,
                                             godot_property_usage_flags usage, godot_property_hint hint, String _hint_string):
    # cdef String _hint_string = String(hint_string)
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

    # print('prop registered', name, __ob_refcnt(property_data))

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


def register_property(type cls, str name, object default_value,
                      godot_method_rpc_mode rpc_mode=GODOT_METHOD_RPC_MODE_DISABLED,
                      godot_property_usage_flags usage=GODOT_PROPERTY_USAGE_DEFAULT,
                      godot_property_hint hint=GODOT_PROPERTY_HINT_NONE,
                      str hint_string=''):
    _register_python_property(cls, name, default_value, rpc_mode, usage, hint, String(hint_string))


ctypedef godot_variant (*__godot_wrapper_method)(godot_object *, void *, void *, int, godot_variant **) nogil

cdef public object _register_python_method(type cls, const char *name, object method, godot_method_rpc_mode rpc_type):
    # Py_INCREF(method)
    cdef godot_instance_method m = [<__godot_wrapper_method>_python_method_wrapper, <void *>method, _python_method_free]
    cdef godot_method_attributes attrs = [rpc_type]

    cdef bytes class_name = cls.__name__.encode('utf-8')
    nsapi.godot_nativescript_register_method(handle, <const char *>class_name, name, attrs, m)


def register_method(type cls, str method_name, object method=None, godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED):
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

        if not (<_Wrapped>self)._owner:
            WARN_PRINT("%r has undefined Godot instance" % self)
            (<_Wrapped>self)._owner = instance  # XXX: tool classes may have _owner set to NULL incorrectly

        # Variant casts from PyObject* are defined in Variant::Variant(const PyObject*) constructor
        return <Variant>method(self, *__parse_args(num_args, args))
