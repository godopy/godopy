cdef extern from "godot_cpp/classes/object.hpp" namespace "godot" nogil:
    cppclass _Object "godot::Object":
        void *_owner

        _Object()
