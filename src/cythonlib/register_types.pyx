cimport godot

# import importlib

'''
cdef int register_PythonModule() except -1:
    PythonModule = godot.GodotExtentionClass('PythonModule', 'Resource')

    def __init__(self):
        self.module = None

    def import_module(self, str name):
        self.module = importlib.import_module(name)

    PythonModule.add_method_to_class(import_module)

    return 0
'''

cdef public int initialize_godopy_python_extension_types(ModuleInitializationLevel p_level) except -1:
    return 0


cdef public int uninitialize_godopy_python_extension_types(ModuleInitializationLevel p_level) except -1:
    return 0
