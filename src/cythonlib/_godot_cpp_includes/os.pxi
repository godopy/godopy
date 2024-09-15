cdef extern from "godot_cpp/classes/os.hpp" namespace "godot" nogil:
    cppclass OS:
        @staticmethod
        OS *get_singleton()

        String read_string_from_stdin()
