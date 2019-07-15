from libc.stdint cimport uint64_t, int64_t
from libc.stddef cimport wchar_t
from libcpp cimport bool

from godot_headers.gdnative_api_struct__gen cimport *

cdef extern from "String.hpp" namespace "godot" nogil:
    cdef cppclass String:
        String() except +
        String(const char *) except +
        String(const String&) except +

        String& operator= (const String&)
        String& operator= (const char*)

        # TODO: Add all String methods by analogy with libcpp.string

# cdef extern from "Variant.hpp" namespace "godot" nogil:
cdef cppclass Variant

cdef extern from "Array.hpp" namespace "godot" nogil:
    cdef cppclass Array:
        godot_array _godot_array
        Array() except+
        Array(const Array &other) except+
        Array()
        void append(const Variant &v)

        @staticmethod
        Array make(...)

cdef extern from "GodotGlobal.hpp" namespace "godot" nogil:
    godot_gdnative_core_api_struct *gdapi "godot::api"
    godot_gdnative_ext_nativescript_api_struct *nativescript_api
    godot_gdnative_ext_nativescript_1_1_api_struct *nativescript_1_1_api

    void *gdnlib

    cdef cppclass Godot:
        @staticmethod
        void print(String &message) except+

        @staticmethod
        void print(const char *message) except+

        @staticmethod
        void print(const char *fmt, ...) except+

    void *_RegisterState__nativescript_handle "godot::_RegisterState::nativescript_handle"
    int _RegisterState__language_index "godot::_RegisterState::language_index"
    int _RegisterState__python_language_index "godot::_RegisterState::python_language_index"

cdef extern from "Vector2.hpp" namespace "godot" nogil:
    # Vector2 is defined as struct in C++
    cdef cppclass Vector2:
        float x, y, w, h # x,w and y,h are unions in C++

        Vector2()
        Vector2(float, float)

cdef extern from "Wrapped.hpp" namespace "godot" nogil:
    cdef cppclass _Wrapped:
        godot_object *_owner
        size_t _type_tag

cdef extern from "Variant.hpp" namespace "godot" nogil:
    cdef cppclass Variant:
        Variant() except+
        Variant(const Variant&) except+
        Variant(bool) except+
        Variant(signed int) except+
        Variant(unsigned int) except+
        Variant(int64_t) except+
        Variant(uint64_t) except+
        Variant(float) except+
        Variant(double) except+
        Variant(const String&) except+
        Variant(const char *) except+
        Variant(const wchar_t *) except+
        Variant(const Vector2&) except+

        Variant &operator=(const Variant &v)
