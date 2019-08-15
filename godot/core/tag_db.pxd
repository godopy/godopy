from godot_headers.gdnative_api cimport godot_object

cdef dict CythonTagDB
cdef dict PythonTagDB
cdef dict __instance_map

cdef register_cython_type(type cls)
cdef register_python_type(type cls)
cdef register_global_cython_type(type cls, str api_name)
cdef register_global_python_type(type cls, str api_name)

cdef get_instance_from_owner(godot_object *instance)
