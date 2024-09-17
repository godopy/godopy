cdef extern from "godot_cpp/classes/os.hpp" namespace "godot" nogil:
    cppclass OS:
        @staticmethod
        OS *get_singleton()

        object read_string_from_stdin()
