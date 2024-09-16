cdef class Class:
    """\
    Represents Godot engine classes. Instances of this class and derived
    classes encapsulate Godot *classes*.
    """
    def __init__(self, str name):
        self.__name__ = name

    def __call__(self):
        return Object(self)


cdef class SingletonClass(Class):
    def __call__(self):
        return Singleton(self)
