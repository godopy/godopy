import godot as gd
import numpy as np

GDExample = gd.ExtensionClass('GDExample', 'Sprite2D')


def __init__(self: gd.Extension):
    print("INIT0")
    self.time_passed = 0.0

    self.set_position = gd.MethodBind(self.__godot_object__, 'set_position')


def _process(self, delta: float) -> None:
    self.time_passed += delta

    new_position = (
        10.0 + (10.0 * np.sin(self.time_passed * 2.0)),
        10.0 + (10.0 * np.cos(self.time_passed * 1.5))
    )

    self.set_position(new_position)


def _ready(self: gd.Extension) -> None:
    gd.print("READY")


GDExample.bind_method(__init__)
GDExample.bind_virtual_method(_process)
GDExample.bind_virtual_method(_ready)
