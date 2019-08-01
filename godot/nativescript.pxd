from godot_headers.gdnative_api cimport godot_method_rpc_mode
from .cpp.core_types cimport Variant as CVariant

cdef extern from "PyGodot.hpp" namespace "pygodot":
    object register_method[M](
        type cls,
        const char *name,
        M method_ptr,
        ...  # default rpc_type
    )

# Fused function type, not used
ctypedef object (*MethodNoArgs)(object)
ctypedef object (*Method__float)(object, const float)

ctypedef fused fusedmethod:
    MethodNoArgs
    Method__float

ctypedef object (*_regfunc_static)()
ctypedef object (*_regfunc_classobj)(type)

ctypedef fused methods_registration_function:
    _regfunc_static
    _regfunc_classobj

cpdef register_class(type cls)

cdef test_method_call(type cls, object instance, fusedmethod method)

cdef _register_class(type cls, methods_registration_function registration_func)
cdef _register_python_method(type cls, const char *name, object method, godot_method_rpc_mode rpc_type=*)
