import godot as gd
import gdextension as gde
import numpy as np

class Sprite2D(gd.EngineClass):
    __godot_class__ = gde.Class._get_class('Sprite2D')

class GDExample(Sprite2D):
    def __init__(self):
        self.time_passed = 0.0

    def _process(self, delta: float) -> None:
        self.time_passed += delta

        new_position = (
            10.0 + (10.0 * np.sin(self.time_passed * 2.0)),
            10.0 + (10.0 * np.cos(self.time_passed * 1.5))
        )

        self.set_position(new_position)

    def _ready(self) -> None:
        gd.print("READY")
