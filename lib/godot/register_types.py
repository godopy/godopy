def initialize(level):
    pass


def uninitialize(level):
    pass


def configure(config):
    from godot.classdb import Object
    from godot.core import _ext_class_cache

    config['object_constructor'] = Object
    config['godot_class_to_class'] = lambda godot_cls: _ext_class_cache[godot_cls]
