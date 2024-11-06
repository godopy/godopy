from godot.classdb import Engine, SceneTree
from godopy.contrib.console import terminal


__all__ = ['_initialize']
__extends__ = SceneTree
__class_name__ = 'TestConsole'


def _initialize(self):
    terminal.interact(Engine.get_version_info())

    self.quit(0)
