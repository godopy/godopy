cdef extern from "godot_cpp/classes/os.hpp" namespace "godot" nogil:
    cdef cppclass OS:
        @staticmethod
        OS *get_singleton()

        String read_string_from_stdin()
        String get_environment(str) const
        bint is_stdout_verbose() const
        uint64_t get_thread_caller_id() const

        PackedStringArray get_cmdline_args()
        PackedStringArray get_cmdline_user_args()
