cdef class PythonCallableBase:
    """
    Base class for BoundExtensionMethod and (TODO) CustomCallable.
    """
    def __cinit__(self, *args):
        self.type_info = ()
        self._type_info_opt = [0]*16
        self.__func__ = None
        self.__name__ = ''


    def __init__(self):
        raise NotImplementedError("Base class, cannot instantiate")


cdef class BoundExtensionMethod(PythonCallableBase):
    def __init__(self, Extension instance, object method, tuple type_info=None):
        self.__name__ = method.__name__
        self.error_count = 0

        if isinstance(method, ExtensionMethod):
            assert method.is_registered, "attempt to bind unregistered extension method"
            self.__func__ = method.__func__
            self.type_info = method.type_info

        elif callable(method):
            self.__func__ = method
            if type_info is None:
                raise TypeError("'type_info' argument is required for python functions")
            self.type_info = type_info

        else:
            raise ValueError("Python callable is required")

        self.__self__ = instance
        make_optimized_type_info(self.type_info, self._type_info_opt)


    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


    def __str__(self):
        return self.__name__


    def __repr__(self):
        return "<%s %s.%s of %r>" % (self.__class__.__name__, self.__self__.__class__.__name__,
                                     self.__name__, self.__self__)


    cdef size_t get_argument_count(self) except -2:
        if self.__func__ is None:
            return -1

        # Don't count self
        return self.__func__.__code__.co_argcount - 1
