cdef dict _METHODDB = {}
cdef dict _CLASSDB = {}


def get_method_info(class_name):
    cdef bytes mi_pickled

    if class_name not in _METHODDB:
        mi_pickled = _global_method_info__pickles.get(class_name, None)
        _METHODDB[class_name] = pickle.loads(mi_pickled) if mi_pickled is not None else {}

    return _METHODDB[class_name] 


cdef class Class:
    """
    Defines all Godot Engine's classes.

    NOTE: Although instances of `gdextension.Class` and its subclasses implement
    class functionality, they are still *objects* on the Python level.

    Only on the higher level (`godot` module) they would be wrapped as real Python
    classes.

    Works as a singleton, can't be instantiated directly: use `Class.get_class`
    in Cython or `Class._get_class` in Python to create/get instances
    `gdextension.Class`.

    Doesn't implement any GDExtension API calls by itself.

    Captures method, property (TODO) and signal (TODO) information,
    processes class inheritance chains.
    """
    def __init__(self, str name):
        raise TypeError("Godot Engine classes can't be instantiated")

    def __call__(self):
        return Object(self)

    def __str__(self):
        return self.__name__

    def __repr__(self):
        class_name = '%s[%s]' % (self.__class__.__name__, self.__name__)
        return "<%s.%s at 0x%016X>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self)

    cdef int initialize_class(self) except -1:
        cdef dict method_info
        cdef str inherits_name

        try:
            method_info = get_method_info(self.__name__)
        except KeyError:
            raise NameError('Class %r not found' % self.__name__)

        self.__method_info__ = method_info

        try:
            inherits_name = _global_inheritance_info[self.__name__]
        except KeyError:
            raise NameError('Parent class of %r not found' % self.__name__)

        if inherits_name is not None:
            self.__inherits__ = Class.get_class(inherits_name)


    cpdef object get_method_info(self, method_name):
        cdef object mi = self.__method_info__.get(method_name, None)

        if mi is None and self.__inherits__ is not None:
            mi = self.__inherits__.get_method_info(method_name)

        return mi


    cdef void *get_tag(self) except NULL:
        cdef PyStringName class_name = PyStringName(self.__name__)

        return gdextension_interface_classdb_get_class_tag(class_name.ptr())


    def get_tag_id(self) -> int:
        return <uint64_t>self.get_tag()


    @staticmethod
    def get(name):
        return Class.get_class(name)


    @staticmethod
    cdef Class get_class(object name):
        cdef Class cls

        if not isinstance(name, str):
            raise ValueError("String is required, got %r" % type(name))

        if name not in _CLASSDB:
            name = str(name)
            sys.intern(name)
            cls = Class.__new__(Class, name)
            _CLASSDB[name] = cls
            cls.__name__ = name
            cls.initialize_class()

        return _CLASSDB[name]
