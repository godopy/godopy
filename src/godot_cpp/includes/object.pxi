cdef extern from "godot_cpp/classes/object.hpp" namespace "godot" nogil:
    cdef cppclass GodotCppObject "godot::Object":
        void *_owner

        GodotCppObject()
        GodotCppObject(void *)

        void set(const StringName &p_property, const Variant &p_value)
        Variant get(const StringName &p_property) const


    cdef GodotCppObject *get_object_instance_binding "godot::internal::get_object_instance_binding" (void *)

ctypedef GodotCppObject * GodotCppObjectPtr
