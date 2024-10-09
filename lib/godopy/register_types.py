import importlib

def python_runtime_class():
    from godot import ExtensionClass
    PythonRuntime = ExtensionClass('PythonRuntime', 'Object')

    @PythonRuntime.bind_method
    def import_module(self, name):
        return importlib.import_module(name)

    return PythonRuntime


def initialize(level):
    # print("Enter godopy initialize %d" % level)

    if level == 0:
        python_runtime_class().register()


def deinitialize(level):
    # print("Enter godopy deinitialize %d" % level)
    pass
