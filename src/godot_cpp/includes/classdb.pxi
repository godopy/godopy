cdef extern from "godot_cpp/classes/class_db_singleton.hpp" namespace "godot" nogil:
    cdef cppclass ClassDBSingleton:
        @staticmethod
        ClassDBSingleton *get_singleton()

        bint class_exists(const StringName &p_class_name) const
        bint can_instantiate(const StringName &p_class_name) const
