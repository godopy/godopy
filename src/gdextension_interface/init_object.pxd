from gdextension_interface cimport *
from cpp cimport ModuleInitializationLevel

ctypedef void (*Callback)(ModuleInitializationLevel)

cdef extern from "godot_cpp/godot.hpp" namespace "godot" nogil:
    cppclass InitObject "godot::GDExtensionBinding::InitObject":
        InitObject()
        InitObject(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization)

        void register_initializer(Callback p_init) const
        void register_terminator(Callback p_init) const
        void set_minimum_library_initialization_level(ModuleInitializationLevel p_level) const

        GDExtensionBool init() const
