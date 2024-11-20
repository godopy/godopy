from typing import Mapping
from godot.classdb import ProjectSettings

def configure(config: Mapping) -> None:
    from godot.classdb import Object
    from godot.core import _ext_class_cache

    config['object_constructor'] = Object
    config['godot_class_to_class'] = lambda godot_cls: _ext_class_cache[godot_cls]

    msg = "Please include this when reporting the bug on: https://github.com/godopy/godopy/issues"
    ProjectSettings.set_setting('debug/settings/crash_handler/message.editor', msg)
