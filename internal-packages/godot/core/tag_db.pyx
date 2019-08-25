from ..globals cimport (
    gdapi,
    gdnlib,
    nativescript_1_1_api as ns11api,
    _nativescript_handle as handle,
    _cython_language_index as CYTHON_IDX,
    _python_language_index as PYTHON_IDX
)

from ..core._meta cimport __tp_dict, PyType_Modified

from ..core._wrapped cimport _Wrapped

cdef dict __CythonTagDB = {}
cdef dict __PythonTagDB = {}
cdef dict __instance_map = {}


cdef register_cython_type(type cls):
    cdef size_t type_tag = <size_t><void *>cls
    cdef bytes name = cls.__name__.encode('utf-8')

    # print('set type tag', name, type_tag)

    ns11api.godot_nativescript_set_type_tag(handle, <const char *>name, <void *>type_tag)

    __tp_dict(cls)['__godot_api_name__'] = cls.__name__
    PyType_Modified(cls)

    __CythonTagDB[type_tag] = cls


cdef register_python_type(type cls):
    cdef size_t type_tag = <size_t><void *>cls
    cdef bytes name = cls.__name__.encode('utf-8')

    ns11api.godot_nativescript_set_type_tag(handle, <const char *>name, <void *>type_tag)

    cls.__godot_api_name__ = cls.__name__

    __PythonTagDB[type_tag] = cls


cdef type get_cython_type(size_t type_tag):
    return __CythonTagDB[type_tag]


cdef type get_python_type(size_t type_tag):
    return __PythonTagDB[type_tag]


cdef register_global_cython_type(type cls, str api_name):
    cdef bytes _api_name = api_name.encode('utf-8')
    cdef size_t type_tag = <size_t><void *>cls

    ns11api.godot_nativescript_set_global_type_tag(CYTHON_IDX, <const char *>_api_name, <const void *>type_tag)

    __tp_dict(cls)['__godot_api_name__'] = api_name
    PyType_Modified(cls)

    __CythonTagDB[type_tag] = cls


cdef register_global_python_type(type cls, str api_name):
    cdef bytes _api_name = api_name.encode('utf-8')
    cdef size_t type_tag = <size_t><void *>cls

    ns11api.godot_nativescript_set_global_type_tag(PYTHON_IDX, <const char *>_api_name, <const void *>type_tag)

    __tp_dict(cls)['__godot_api_name__'] = api_name
    PyType_Modified(cls)

    __PythonTagDB[type_tag] = cls


cdef register_godot_instance(godot_object *godot_instance, object python_instance):
    __instance_map[<size_t>godot_instance] = python_instance


cdef unregister_godot_instance(godot_object *godot_instance):
    cdef size_t godot_instance_tag = <size_t>godot_instance
    if godot_instance_tag in __instance_map:
        del __instance_map[godot_instance_tag]


cdef get_python_instance(godot_object *godot_instance):
    return __instance_map[<size_t>godot_instance]


cdef replace_python_instance(godot_object *godot_instance, object python_instance):
    cdef size_t godot_instance_tag = <size_t>godot_instance
    cdef _Wrapped old_instance = <_Wrapped>__instance_map.get(godot_instance_tag, None)

    if old_instance is not None:
        old_instance._owner = NULL

    __instance_map[godot_instance_tag] = python_instance


cdef bint is_godot_instance_registered(size_t godot_instance_tag) except -1:
    return godot_instance_tag in __instance_map


cdef clear_cython():
    __CythonTagDB.clear()

cdef clear_python():
    __PythonTagDB.clear()

cdef clear_instance_map():
    __instance_map.clear()
