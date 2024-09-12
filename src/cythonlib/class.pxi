cdef class GodotClass:
    """\
    Represents Godot engine classes. Instances of this class and derived
    classes encapsulate Godot *classes*.
    """
    def __init__(self, str name):
        self.name = name

    def __call__(self):
        return GodotObject(self.name)


cdef class GodotSingletonClass(GodotClass):
    def __call__(self):
        return GodotSingleton(self.name)
