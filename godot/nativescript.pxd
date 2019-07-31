from godot_headers.gdnative_api cimport godot_method_rpc_mode

ctypedef object (*MethodNoArgs)(object)
ctypedef object (*Method__float)(object, const float)

ctypedef fused fusedmethod:
    MethodNoArgs
    Method__float

cpdef register_class(type cls)
cdef register_method(type cls, str name, fusedmethod method, godot_method_rpc_mode rpc_type=*)

cdef test_method_call(type cls, object instance, fusedmethod method)
