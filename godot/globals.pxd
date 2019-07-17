from .headers.gdnative_api cimport *
from .core_cctypes cimport String

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

    void *_nativescript_handle "godot::_RegisterState::nativescript_handle"
    int _language_index "godot::_RegisterState::language_index"
    int _python_language_index "godot::_RegisterState::python_language_index"
