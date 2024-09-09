cdef extern from "godot_cpp/variant/utility_functions.hpp" namespace "godot" nogil:
    cdef cppclass UtilityFunctions:
        @staticmethod
        void print(const char *)

        @staticmethod
        void print_verbose(const char *)

        @staticmethod
        void print_rich(const char *)

        @staticmethod
        void printerr(const char *)

        @staticmethod
        void printraw(const char *)

        @staticmethod
        void push_error(const char *)
