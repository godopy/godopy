cdef extern from "godot_cpp/variant/string_name.hpp" namespace "godot" nogil:
    cppclass StringName:
        StringName()
        StringName(const char *)
        StringName(const char *, bint)
        StringName(str)
        StringName(bytes)

        void *_native_ptr()
        bytes py_bytes()
        str py_str()
