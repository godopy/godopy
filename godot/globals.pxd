from godot_headers.gdnative_api cimport (
    godot_gdnative_core_api_struct,
    godot_gdnative_ext_nativescript_api_struct,
    godot_gdnative_ext_nativescript_1_1_api_struct,
    godot_gdnative_ext_pluginscript_api_struct
)
from .cpp.core_types cimport String


cdef extern from "Defs.hpp":
    cdef void WARN_PRINT(const char *msg)
    cdef const char *WARN_PRINTS(const char *msg)
    cdef void ERR_PRINT(const char *msg)
    cdef const char *ERR_PRINTS(const char *msg)
    cdef void FATAL_PRINT(const char *msg)
    cdef void ERR_FAIL_INDEX(int index, int size)
    cdef void CRASH_BAD_INDEX(int index, int size)
    cdef void ERR_FAIL_NULL(void *param)
    cdef void ERR_FAIL_COND(bint cond)
    cdef void CRASH_COND(bint cond)
    cdef void CRASH_NOW()


cdef extern from "GodotGlobal.hpp" namespace "godot" nogil:
    godot_gdnative_core_api_struct *gdapi "godot::api"
    godot_gdnative_ext_nativescript_api_struct *nativescript_api
    godot_gdnative_ext_nativescript_1_1_api_struct *nativescript_1_1_api
    godot_gdnative_ext_pluginscript_api_struct *pluginscript_api

    void *gdnlib

    cdef cppclass Godot:
        @staticmethod
        void print(String &message)

        @staticmethod
        void print(const String &fmt, ...)

    void *_nativescript_handle "godot::_RegisterState::nativescript_handle"
    int _language_index "godot::_RegisterState::language_index"
    int _cython_language_index "godot::_RegisterState::cython_language_index"
    int _python_language_index "godot::_RegisterState::python_language_index"
