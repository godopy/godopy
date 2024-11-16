from cpython cimport PyObject
from libc.stddef cimport wchar_t

cdef extern from "godot_cpp/variant/char_string.hpp" namespace "godot" nogil:
    cdef cppclass CharString:
        const char *get_data()
        int resize(int64_t p_size)
        const char *ptr()
        char *ptrw()
        void set(int64_t p_index, const char &p_elem)
        void set_zero(int64_t p_index)

    cdef cppclass CharWideString:
        const wchar_t *get_data()
        int resize(int64_t p_size)
        const wchar_t *ptr()
        wchar_t *ptrw()
        void set(int64_t p_index, const wchar_t &p_elem)
        void set_zero(int64_t p_index)


cdef extern from "godot_cpp/variant/string.hpp" namespace "godot" nogil:
    cdef cppclass String:
        String()
        String(const char *)
        String(const wchar_t *)
        String(const StringName &)
        String(const NodePath &)
        String(const PyObject *)
        bint operator==(const String&)
        bint operator==(const char *)
        bint operator!=(const String&)
        bint operator!=(const char *)

        CharString utf8()

        String get_base_dir() const

        void *_native_ptr()
