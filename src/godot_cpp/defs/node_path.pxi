cdef extern from "godot_cpp/variant/node_path.hpp" namespace "godot" nogil:
    cdef cppclass String

    cppclass NodePath:
        NodePath()
        NodePath(const char *)
        NodePath(str)
        NodePath(bytes)
        NodePath(const String&)

        void *_native_ptr()

        bytes py_bytes()
        str py_str()
