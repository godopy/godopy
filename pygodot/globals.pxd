from godot_headers.gdnative_api cimport *
from .cctypes cimport String

# TODO: Add ERR/WARN macros!

cdef extern from "GodotGlobal.hpp" namespace "godot" nogil:
    godot_gdnative_core_api_struct *gdapi "godot::api"
    godot_gdnative_ext_nativescript_api_struct *nativescript_api
    godot_gdnative_ext_nativescript_1_1_api_struct *nativescript_1_1_api

    void *gdnlib
    bint in_editor

    cdef cppclass Godot:
        @staticmethod
        void print(String &message)

        @staticmethod
        void print(const char *message)

        @staticmethod
        void print(const char *fmt, ...)

    void *_nativescript_handle "godot::_RegisterState::nativescript_handle"
    int _language_index "godot::_RegisterState::language_index"
    int _python_language_index "godot::_RegisterState::python_language_index"
