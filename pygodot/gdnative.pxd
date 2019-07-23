from godot_headers.gdnative_api cimport godot_method_rpc_mode, GODOT_METHOD_RPC_MODE_DISABLED

cpdef object register_class(object cls)
cpdef object register_method(object cls, object method, godot_method_rpc_mode rpc_type=?)
