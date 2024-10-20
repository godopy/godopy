cdef extern from "godot_cpp/classes/object.hpp" namespace "godot" nogil:
    cdef cppclass GodotCppObject "godot::Object":
        void *_owner

        GodotCppObject()
        GodotCppObject(void *)

    cdef GodotCppObject *get_object_instance_binding "godot::internal::get_object_instance_binding" (void *)

ctypedef GodotCppObject * GodotCppObjectPtr
