cdef extern from "godot_cpp/classes/engine.hpp" namespace "godot" nogil:
    cdef cppclass Engine:
        @staticmethod
        Engine *get_singleton()

        bint has_singleton(str) const
        bint is_editor_hint() const
