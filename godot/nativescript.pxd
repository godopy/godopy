cpdef register_class(type cls)

cdef extern from "PyGodot.hpp" namespace "pygodot":
    void register_method[M](
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

cdef test_method_call(type cls, object instance, fusedmethod method)
