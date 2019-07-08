from godot.gdnative_api_struct__gen cimport *

cdef extern from "String.hpp" namespace "godot" nogil:
    cdef cppclass String:
        String() except +
        String(const char *) except +
        String(const String&) except +

        String& operator= (const String&)
        String& operator= (const char*)

        # TODO: Add all String methods by analogy with libcpp.string

cdef extern from "GodotGlobal.hpp" namespace "godot" nogil:
    godot_gdnative_core_api_struct *gdapi "godot::api"
    godot_gdnative_ext_nativescript_api_struct *nativescript_api
    godot_gdnative_ext_nativescript_1_1_api_struct *nativescript_1_1_api

    void *gdnlib

    cdef cppclass Godot:
        @staticmethod
        void print(String &message)
