cdef dict _method_info_cache = {}
def get_method_info(class_name):
    cdef bytes mi_pickled

    if class_name not in _method_info_cache:
        mi_pickled = _global_method_info__pickles[class_name]
        _method_info_cache[class_name] = pickle.loads(mi_pickled)

    return _method_info_cache[class_name] 


cdef dict _class_cache = {}

cdef class Class:
    """\
    Represents Godot engine classes. Instances of this class and derived
    classes encapsulate Godot *classes*.
    """
    def __init__(self, str name):
        raise TypeError("Godot Engine classes can't be instantiated")

    def __call__(self):
        return Object(self)

    cdef int initialize_class(self) except -1:
        try:
            self._methods = get_method_info(self.__name__)
        except KeyError:
            raise NameError('Class %r not found' % self.__name__)


    @staticmethod
    cdef Class get_class(str name):
        cdef Class cls
        if name not in _class_cache:
            cls = Class.__new__(Class, name)
            _class_cache[name] = cls
            cls.__name__ = name
            cls.initialize_class()

        return _class_cache[name]
