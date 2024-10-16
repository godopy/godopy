cdef extern from "godot_cpp/classes/project_settings.hpp" namespace "godot" nogil:
    cppclass ProjectSettings:
        @staticmethod
        ProjectSettings *get_singleton()

        bint has_setting(str)
