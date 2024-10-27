cdef class EngineCallableBase:
    """
    Base class for MethodBind, UtilityFunction and BuiltinMethod.
    """
    def __cinit__(self, *args):
        self.type_info = ()
        self._type_info_opt = [0]*16
        self.__name__ = ''

    def __init__(self):
        raise NotImplementedError("Base class, cannot instantiate")
