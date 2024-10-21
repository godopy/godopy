cdef extern from "godot_cpp/variant/rid.hpp" namespace "godot" nogil:
    cdef cppclass _RID "godot::RID":
        _RID()

        int64_t get_id()
