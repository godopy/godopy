cdef extern from "godot_cpp/classes/engine.hpp" namespace "godot" nogil:
    cppclass Engine:
        @staticmethod
        Engine *get_singleton()

        bint has_singleton(str) const
