import godot as gd
from godot.extension import ExtensionClass, Extension
from godot import MethodBind

import numpy as np

GDExample = ExtensionClass('GDExample', 'Sprite2D')


def __init__(self: Extension):
    print("INIT0")
    self.time_passed = 0.0


def _process(self, delta: float) -> None:
    self.time_passed += delta
    new_position = (
        10.0 + (10.0 * np.sin(self.time_passed * 2.0)),
        10.0 + (10.0 * np.cos(self.time_passed * 1.5))
    )

    # set_position = MethodBind(self, 'set_position')
    # set_position(new_position)


def _ready(self: Extension) -> None:
    gd.print("READY")


GDExample.bind_method(__init__)
GDExample.bind_virtual_method(_process)
GDExample.bind_virtual_method(_ready)
