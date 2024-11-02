from cpython cimport PyObject

cdef extern from "godot_cpp/variant/string_name.hpp" namespace "godot" nogil:
    cdef cppclass String

    cdef cppclass StringName:
        StringName()
        StringName(const char *)
        StringName(const char *, bint)
        StringName(const PyObject *)
        StringName(const String&)
        StringName(const StringName&)
        bint operator==(const StringName&)
        bint operator==(const String&)
        bint operator==(const char *)
        bint operator!=(const StringName&)
        bint operator!=(const String&)
        bint operator!=(const char *)

        void *_native_ptr()
