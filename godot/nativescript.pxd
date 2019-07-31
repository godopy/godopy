cdef extern from "PyGodot.hpp" namespace "pygodot":
    void _register_method[M](
        const char *class_name,
        const char *name,
        M method_ptr,
        ...  # default rpc_type
    )

cdef extern from "PyGodot.hpp" namespace "godot":
    void register_method[M](
        type cls,
        const char *name,
        M method_ptr,
        ... # default rpc_type
    )


# Fused function type, not used
ctypedef object (*MethodNoArgs)(object)
ctypedef object (*Method__float)(object, const float)

ctypedef fused fusedmethod:
    MethodNoArgs
    Method__float


cpdef register_class(type cls)

cdef test_method_call(type cls, object instance, fusedmethod method)
