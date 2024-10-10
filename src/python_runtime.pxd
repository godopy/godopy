cdef extern from "python/python_object.h" nogil:
    cppclass PythonObject:
        void *_owner

        PythonObject()

        void init_ref()
        void unreference()


cdef extern from "python/python_runtime.h" nogil:
    cppclass PythonRuntime:
        @staticmethod
        PythonRuntime *get_singleton()

        PythonObject *python_object_from_pyobject(object)
