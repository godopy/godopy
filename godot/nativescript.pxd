from godot_headers.gdnative_api cimport (
    godot_method_rpc_mode, godot_property_usage_flags, godot_property_hint
)
from .cpp.core_types cimport String

ctypedef object (*_regfunc_static)()
ctypedef object (*_regfunc_classobj)(type)

ctypedef fused methods_registration_function:
    _regfunc_static
    _regfunc_classobj

cpdef register_class(type cls)
cdef _register_class(type cls, methods_registration_function registration_func)

cdef extern from "PyGodot.hpp" namespace "pygodot":
    object register_method[M](
        type cls,
        const char *name,
        M method_ptr,
        ...  # default rpc_type
    )

cdef _register_python_method(type cls, const char *name, object method, godot_method_rpc_mode rpc_type=*)

cdef register_property(
    type cls, const char *name, object default_value, godot_method_rpc_mode rpc_mode=*,
    godot_property_usage_flags usage=*, godot_property_hint hint=*, str hint_string=*
)

cdef register_signal(type cls, str name, object args=*)
