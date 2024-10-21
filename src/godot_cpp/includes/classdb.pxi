cdef extern from "godot_cpp/classes/class_db_singleton.hpp" namespace "godot" nogil:
    cdef cppclass ClassDB "godot::ClassDBSingleton":
        @staticmethod
        ClassDB *get_singleton()

        bint class_exists(str p_class_name) const
        bint is_parent_class(str p_class_name, str p_inherits) const
        bint can_instantiate(str p_class_name) const
        bint class_has_method(str p_class_name, str p_method) const
        bint class_has_method(str p_class_name, str p_method, bint p_no_inheritance) const
        int32_t class_get_method_argument_count(str p_class, str p_method) const
        int32_t class_get_method_argument_count(str p_class, str p_method, bint p_no_inheritance) const
        bint is_class_enabled(str p_class) const
        Array class_get_method_list(str p_class) const
        Array class_get_method_list(str p_class, bint p_no_inheritance) const
