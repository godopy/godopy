cdef class Class:
    """\
    Represents Godot engine classes. Instances of this class and derived
    classes encapsulate Godot *classes*.
    """
    def __init__(self, str name):
        self.__name__ = name
        try:
            self._methods = pickle.loads(_method_data[name])
        except KeyError:
            raise NameError('Class %r not found' % name)

    def __call__(self):
        return Object(self)
