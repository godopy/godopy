cdef extern from "godot_cpp/variant/rid.hpp" namespace "godot" nogil:
    cdef cppclass _RID "godot::RID":
        _RID()
        _RID(const _RID &)

        int64_t get_id()
        bint is_valid()
