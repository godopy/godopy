cdef extern from "godot_cpp/variant/string_name.hpp" namespace "godot" nogil:
    cppclass StringName:
        StringName()
        StringName(const char *)
        StringName(const char *, bint)

        void *_native_ptr()
