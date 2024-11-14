from godot.classdb import SceneTree
from godopy.contrib.console import terminal


__all__ = ['_initialize']
__extends__ = SceneTree
__class_name__ = 'TestConsole'


def _initialize(self):
    terminal.interact()

    self.quit(0)
