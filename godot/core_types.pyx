from .globals cimport (
    gdapi,
    nativescript_1_1_api as ns11api,
    _nativescript_handle as handle,
    _cython_language_index as cython_idx,
    _python_language_index as python_idx
)

cdef class _Wrapped:
    pass

cdef class _PyWrapped:
    pass

cdef dict CythonTagDB = {}
cdef dict PythonTagDB = {}

cdef register_cython_type(type cls):
    cdef size_t type_tag = <size_t><void *>cls
    cdef bytes name = cls.__name__.encode('utf-8')

    ns11api.godot_nativescript_set_type_tag(handle, <const char *>name, <void *>type_tag)

    CythonTagDB[type_tag] = cls


cdef register_python_type(type cls):
    cdef size_t type_tag = <size_t><void *>cls
    cdef bytes name = cls.__name__.encode('utf-8')

    ns11api.godot_nativescript_set_type_tag(handle, <const char *>name, <void *>type_tag)

    PythonTagDB[type_tag] = cls


cdef register_global_cython_type(type cls, str api_name):
    cdef bytes _api_name = api_name.encode('utf-8')
    cdef size_t type_tag = <size_t><void *>cls

    ns11api.godot_nativescript_set_global_type_tag(cython_idx, <const char *>_api_name, <const void *>type_tag)

    CythonTagDB[type_tag] = cls


cdef register_global_python_type(type cls, str api_name):
    cdef bytes _api_name = api_name.encode('utf-8')
    cdef size_t type_tag = <size_t><void *>cls

    ns11api.godot_nativescript_set_global_type_tag(python_idx, <const char *>_api_name, <const void *>type_tag)

    PythonTagDB[type_tag] = cls

    # cls.__godot_api_name__ = api_name
