cdef extern from "godot_cpp/variant/string_name.hpp" namespace "godot" nogil:
    cdef cppclass String

    cppclass StringName:
        StringName()
        StringName(const char *)
        StringName(const char *, bint)
        StringName(str)
        StringName(bytes)
        StringName(const String&)
        StringName(const StringName&)
        bint operator==(const StringName&)
        bint operator==(const String&)
        bint operator==(const char *)
        bint operator!=(const StringName&)
        bint operator!=(const String&)
        bint operator!=(const char *)

        void *_native_ptr()
        void *ptr "_native_ptr" ()
        bytes py_bytes()
        str py_str()
