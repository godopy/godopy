from cpython cimport PyObject
from libc.stddef cimport wchar_t

cdef extern from "godot_cpp/variant/char_string.hpp" namespace "godot" nogil:
    cdef cppclass CharString:
        const char *get_data()

cdef extern from "godot_cpp/variant/string.hpp" namespace "godot" nogil:
    cppclass String:
        String()
        String(const char *)
        String(const wchar_t *)
        String(str)
        String(bytes)
        String(const PyObject *)
        bint operator==(const String&)
        bint operator==(const char *)
        bint operator!=(const String&)
        bint operator!=(const char *)

        CharString utf8()
        bytes py_bytes()
        str py_str()
