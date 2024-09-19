from godot.extension import ExtensionClass
from godot import MethodBind

import numpy as np

GDExample = ExtensionClass('GDExample', 'Sprite2D')

def __init__(self):
    print("INIT0")
    self.time_passed: float = 0.0

def _process(self, delta: float) -> None:
    print("PROCESS")
    self.time_passed += delta
    new_position = (
        10.0 + (10.0 * np.sin(self.time_passed * 2.0)),
        10.0 + (10.0 * np.cos(self.time_passed * 1.5))
    )

    set_position = MethodBind(self, 'set_position')
    set_position(new_position)

GDExample.add_methods(__init__, _process)
