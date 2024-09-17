cdef extern from "godot_cpp/classes/os.hpp" namespace "godot" nogil:
    cppclass OS:
        @staticmethod
        OS *get_singleton()

        str read_string_from_stdin()
        str get_environment(str)
