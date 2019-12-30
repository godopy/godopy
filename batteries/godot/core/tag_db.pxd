from godot_headers.gdnative_api cimport godot_object

# cdef dict CythonTagDB
# cdef dict PythonTagDB
# cdef dict __instance_map

cdef register_cython_type(type cls)
cdef register_python_type(type cls)

cdef type get_cython_type(size_t type_tag)
cdef type get_python_type(size_t type_tag)

cdef register_global_cython_type(type cls, str api_name)
cdef register_global_python_type(type cls, str api_name)

cdef register_godot_instance(godot_object *godot_instance, object python_instance)
cdef unregister_godot_instance(godot_object *godot_instance)

cdef replace_python_instance(godot_object *godot_instance, object python_instance)
cdef get_python_instance(godot_object *godot_instance)

cdef bint is_godot_instance_registered(size_t godot_instance_tag) except -1

cdef clear_cython()
cdef clear_python()
cdef clear_instance_map()

