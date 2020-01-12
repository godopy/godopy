from godot_headers.gdnative_api cimport (
    godot_gdnative_core_api_struct,
    godot_gdnative_core_1_1_api_struct,
    godot_gdnative_ext_nativescript_api_struct,
    godot_gdnative_ext_nativescript_1_1_api_struct,
    godot_gdnative_ext_pluginscript_api_struct
)

from .core cimport cpp_types as cpp


cdef extern from "Defs.hpp":
    cdef void WARN_PRINT(str msg)
    cdef void WARN_PRINTS(const char *msg)
    cdef void ERR_PRINT(str msg)
    cdef void ERR_PRINTS(const char *msg)
    cdef void FATAL_PRINT(str msg)
    cdef void ERR_FAIL_INDEX(int index, int size)
    cdef void CRASH_BAD_INDEX(int index, int size)
    cdef void ERR_FAIL_NULL(void *param)
    cdef void ERR_FAIL_PYTHON_NULL(object param)
    cdef object ERR_FAIL_PYTHON_NULL_V(object param, object ret)
    cdef void ERR_FAIL_COND(bint cond)
    cdef object ERR_FAIL_COND_V(bint cond, object ret)
    cdef void CRASH_COND(bint cond)
    cdef void CRASH_NOW()


cdef extern from "GodotGlobal.hpp" namespace "godot" nogil:
    godot_gdnative_core_api_struct *gdapi "godot::api"
    godot_gdnative_core_1_1_api_struct *core_1_1_api
    godot_gdnative_ext_nativescript_api_struct *nativescript_api
    godot_gdnative_ext_nativescript_1_1_api_struct *nativescript_1_1_api
    godot_gdnative_ext_pluginscript_api_struct *pluginscript_api

    void *gdnlib

    cdef cppclass Godot:
        @staticmethod
        void print(cpp.String &message)

        @staticmethod
        void print(const cpp.String &fmt, ...)

        @staticmethod
        void print(str)

        @staticmethod
        void print(str, ...)


    void *_nativescript_handle "godot::_RegisterState::nativescript_handle"
    int _language_index "godot::_RegisterState::language_index"
    int _cython_language_index "godot::_RegisterState::cython_language_index"
    int _python_language_index "godot::_RegisterState::python_language_index"


# cdef extern from "PythonGlobal.hpp":
#     cdef void PYGODOT_CHECK_NUMPY_API()


cdef extern from "PythonGlobal.hpp" namespace "godopy":
    cdef cppclass GodoPy:
        @staticmethod
        void numpy_init()

        @staticmethod
        void set_cython_language_index(int)

        @staticmethod
        void set_python_language_index(int)
