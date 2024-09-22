# cython: c_string_type=unicode, c_string_encoding=UTF8

from godot_cpp cimport *

include "api_data.pxi"

cdef public int entry_symbol_hook(InitObject *p_init_object) except -1 nogil:
    # Enough for now

    # cdef InitObject init_object = InitObject(p_get_proc_address, p_library, r_initialization)
    # init_object.init()

    # cpp.UtilityFunctions.print_rich("[color=yellow]Hook is active![/color]")

    return 0


# Do this later
cdef int real_entry_symbol_hook(InitObject *p_init_object) except -1:
    cdef GDExtensionBinding binding = GDExtensionBinding.from_entry_symbol(p_init_object)
        # GDExtensionBinding.from_entry_symbol(p_get_proc_address, p_library, r_initialization)

    try:
        import gdextension_interface
    except ImportError:
        import sys
        sys.stderr.write("\nCould not import GDExtension entry hook, aborting\n\n")
        sys.exit(-1)
    
    gdextension_interface.gdextension_entry_hook(binding)

    return 0


cdef object python_initialize_level = None
cdef object python_deinitialize_level = None

cdef void initialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    with gil:
        if python_initialize_level is not None and callable(python_initialize_level):
            python_initialize_level(p_level)

cdef void deinitialize_level(ModuleInitializationLevel p_level) noexcept nogil:
    with gil:
        if python_deinitialize_level is not None and callable(python_deinitialize_level):
            python_deinitialize_level(p_level)


cdef class GDExtensionBinding:
    cdef InitObject *init_object
    cdef ModuleInitializationLevel minimum_initialization_level

    def __cinit__(self):
        self.init_object = NULL
        self.minimum_initialization_level = ModuleInitializationLevel.MODULE_INITIALIZATION_LEVEL_CORE

    @staticmethod
    cdef from_entry_symbol(InitObject *p_init_object):
        cdef GDExtensionBinding binding = GDExtensionBinding.__new__(GDExtensionBinding)

        binding.init_object = p_init_object

        binding.init_object.register_initializer(initialize_level)
        binding.init_object.register_terminator(deinitialize_level)
    
    def register_initializer(self, func):
        # TODO: Check func
        python_initialize_level = func

    def register_uninitializer(self, func):
        # TODO: Check func
        python_uninitialize_level = func

    def set_minimum_library_initialization_level(self, ModuleInitializationLevel p_level):
        self.minimum_initialization_level = p_level
        self.init_object.set_minimum_library_initialization_level(p_level)

    def init(self):
        self.init_object.init()
