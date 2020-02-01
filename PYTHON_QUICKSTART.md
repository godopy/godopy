# Pure Python API

## Getting started

### Prerequisites

It is required that your Godot 3.2 binary be in your [PATH](https://docs.godotengine.org/en/3.2/getting_started/editor/command_line_tutorial.html#path)

TODO: Describe Python 3 requirement

You can verify that Python and Godot are installed correctly by typing **python** and **godot** from your shell.

You should see something like this:
```
$ python --version
Python 3.8.1

$ godot --version
3.2.stable
```

### Setting up a new project

```
$ mkdir godopy-python-example
$ cd godopy-python-example
$ python3 -m venv tools
$ source tools/bin/activate  # `.\tools\Scripts\activate` on Windows
(tools) pip install godopy  # or `pip install path/to/cusomized/GodoPy`
(tools) godopy enable-runpy
```
> Read `BUILD_*.md` to build your own version whith any Python dependencies you want

### Creating a simple GodoPy Python scipt

Create `example.py`:
```py
import os
import math
import numpy as np

from godot import bindings
from godot.globals import gdclass, gdprint
from godot.nativescript import register_class, gdmethod, gdexport, gdsignal
from godot.core.signals import SignalArgumentObject as SAO, SignalArgumentVector2 as SAV2

with open(os.path.join(os.path.dirname(__file__), 'data', 'godopy-logo.png'), 'rb') as fp:
    IMAGE_DATA = np.frombuffer(fp.read(), dtype=np.uint8)


@gdclass
class Icon(bindings.Sprite):
    amplitude = gdexport(10)
    position_changed = gdsignal(SAO('node'), SAV2('new_position'))

    def _init(self):
        image = bindings.Image()

        # In 'runpy' mode there is no real Godot project and `image.load('res://data/godopy-logo.png')` won't work
        # But, if the proper Godot project was available, `image.load(path)` could be used
        image.load_png_from_buffer(IMAGE_DATA)

        self.texture = bindings.ImageTexture()
        self.texture.create_from_image(image)

        self.centered = False
        self.time_emit = 0
        self.time_passed = 0.0
        self.amplitude = 10
        self._position = np.array([0, 0], dtype=np.float32)

    @gdmethod
    def _process(self, delta):
        self.time_passed += delta

        self._position[0] = self.amplitude + (self.amplitude * math.sin(self.time_passed * 2.0))
        self._position[1] = self.amplitude + (self.amplitude * math.cos(self.time_passed * 1.5))

        self.position = self._position  # Vector2 instance would also work

        self.time_emit += delta

        if self.time_emit >= 2:
            self.emit_signal('position_changed', self, self.position)
            self.time_emit = 0


@gdclass
class Example(bindings.Node2D):
    def _init(self):
        self.icon = Icon()
        self.add_child(self.icon)

    @gdmethod
    def _ready(self):
        self.icon.connect('position_changed', self, '_on_position_changed')

    @gdmethod
    def _on_position_changed(self, node, new_position):
        gdprint('{0} {1}', node, new_position)


@gdclass(bindings.SceneTree, '_init')
def Main(self):
    self.example = Example()
    self.get_root().add_child(self.example)


def _init():
    register_class(Icon)
    register_class(Example)
    register_class(Main)
```

Download the [sample logo](https://github.com/godopy/godopy-python-examples/blob/master/2d/simple/data/godopy-logo.png)
(or use any other image) and put it in the folder `data`


Now we can run the example:

```
$ godopy runpy example
```

There are more examples at the [GodoPy examples repo](https://github.com/godopy/godopy-python-examples)

TODO: Describe `project.py`, `installscripts`, `run` and `runeditor` commands, change the code to use `image.load(path)`
