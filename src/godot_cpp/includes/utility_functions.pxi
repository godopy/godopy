cdef extern from "godot_cpp/variant/utility_functions.hpp" namespace "godot" nogil:
    cdef cppclass UtilityFunctions:
        @staticmethod
        void print(str)

        @staticmethod
        void print(const Variant &)

        @staticmethod
        void print(const String &)

        @staticmethod
        void print_verbose(str)

        @staticmethod
        void printerr(str)

        @staticmethod
        void printraw(str)

        @staticmethod
        void print_rich(str)

        @staticmethod
        void push_error(str)

        @staticmethod
        void push_warning(str)
