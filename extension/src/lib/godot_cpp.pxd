cdef extern from "godot_cpp/variant/utility_functions.hpp" namespace "godot" nogil:
    cdef cppclass UtilityFunctions:
        @staticmethod
        void print(const char *)

        @staticmethod
        void printerr(const char *)
