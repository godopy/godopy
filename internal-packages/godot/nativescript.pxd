from godot_headers.gdnative_api cimport (
    godot_method_rpc_mode, godot_property_usage_flags, godot_property_hint
)

cdef extern from "Godot.hpp" namespace "godot":
    void register_cpp_class "godot::register_class" [T] ()

cpdef register_class(type cls)

cdef extern from "PyGodot.hpp" namespace "pygodot":
    object register_method[M](
        type cls,
        const char *name,
        M method_ptr,
        ...  # default rpc_type
    )

    object register_property(
        type cls,
        const char *name,
        object default_value,
        ... # defaults: rpc_mode, usage, hint, hint_string
    )

    object register_signal(
        type cls,
        str name,
        tuple args
    )
