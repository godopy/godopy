from godot_headers.gdnative_api cimport godot_method_rpc_mode
from .core_types cimport _Wrapped

cpdef register_class(type cls)
cpdef register_method(type cls, str name, object method=*, godot_method_rpc_mode rpc_type=*)

cdef test_method_call(type cls, object instance, object method)
