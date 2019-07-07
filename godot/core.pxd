cdef extern from "String.hpp" namespace "godot" nogil:
    cdef cppclass String:
        String() except +
        String(const char *) except +
        String(const String&) except +

        String& operator= (const String&)
        String& operator= (const char*)

        # TODO: Add all String methods by analogy with libcpp.string

cdef extern from "GodotGlobal.hpp" namespace "godot" nogil:
    cdef cppclass Godot:
        @staticmethod
        void print(String &message)
