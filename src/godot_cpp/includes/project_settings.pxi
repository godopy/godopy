cdef extern from "godot_cpp/classes/project_settings.hpp" namespace "godot" nogil:
    cdef cppclass ProjectSettings:
        @staticmethod
        ProjectSettings *get_singleton()

        bint has_setting(const String &)
        bint has_setting(object)
