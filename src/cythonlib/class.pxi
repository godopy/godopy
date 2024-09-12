cdef class GodotClass:
    """\
    Represents Godot engine classes. Instances of this class and derived
    classes encapsulate Godot *classes*.
    """
    # cdef str name
    # cdef StringName _name

    def __init__(self, str name):
        self.name = name
        self._name = stringname_from_str(name)

    def __call__(self):
        return GodotObject(self.name)


cdef class GodotSingletonClass(GodotClass):
    def __call__(self):
        return GodotSingleton(self.name)
