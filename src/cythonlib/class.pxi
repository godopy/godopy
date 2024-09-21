cdef dict _global_inheritance_info = pickle.loads(_global_inheritance_info__pickle)

cdef dict _method_info_cache = {}
def get_method_info(class_name):
    cdef bytes mi_pickled

    if class_name not in _method_info_cache:
        mi_pickled = _global_method_info__pickles[class_name]
        _method_info_cache[class_name] = pickle.loads(mi_pickled)

    return _method_info_cache[class_name] 


DEF ROOT_CLASS = 'Object'
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

    cdef int initialize_class(self, dict opts) except -1:
        cdef dict method_info
        cdef str inherits_name

        if opts.get('skip_method_check'):
            method_info = {}
        else:
            try:
                method_info = get_method_info(self.__name__)
            except KeyError:
                raise NameError('Class %r not found' % self.__name__)

        if self.__name__ != ROOT_CLASS:
            try:
                inherits_name = _global_inheritance_info[self.__name__]
            except KeyError:
                raise NameError('Parent class of %r not found' % self.__name__)

            self.__inherits__ = Class.get_class(inherits_name, {})
        else:
            self.__inherits__ = None

        self.__method_info__ = method_info

    cdef dict get_method_info(self, method_name):
        cdef dict mi = None
        if self.__name__ == ROOT_CLASS:
            return self.__method_info__.get(method_name, None)

        return self.__method_info__.get(
            method_name,
            self.__inherits__.get_method_info(method_name)
        )

    @staticmethod
    cdef Class get_class(str name, dict opts):
        cdef Class cls
        if name not in _class_cache:
            cls = Class.__new__(Class, name)
            _class_cache[name] = cls
            cls.__name__ = name
            cls.initialize_class(opts)

        return _class_cache[name]
