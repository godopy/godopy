# Pure Python API

**DOES NOT WORK YET**

The pure Python API is differs from the Cython API in the following ways:
* Compilation is not required
* Python wrappers should be used in place of native C++ classes

## Getting started

### Prerequisites

[**Build latest version of Godot**](https://godot.readthedocs.io/en/latest/development/compiling/index.html)

TODO: Describe Python 3 requirement

### Setting up a new project

The instructions below assume using git for managing your project.

```
$ mkdir gdnative-python-example
$ cd gdnative-python-example
$ git clone --recursive https://github.com/ivhilaire/pygodot
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/ivhilaire/pygodot
$ git submodule update --init --recursive
```

Build PyGodot and set up a development environment for your platform:
- [MacOS](BUILD_MACOS.md)
- [Linux](BUILD_LINUX.md)
- [Windows](BUILD_WINDOWS.md)

Simple `pip install pygodot` setup will be available in the future versions.

### Creating pure Python GDNative extension

We’ll start by creating an empty Godot project in which we’ll place a few files.

Open Godot and create a new project. For this example, we will place it in a folder called `demo` inside our PyGodot project’s folder structure.

In our demo project, we’ll create a scene containing a Node2D called “Main” and we’ll save it as `main.tscn`.
We’ll come back to that later.

Back in the top-level project folder, we’re also going to create a subfolder called `_demo`
in which we’ll place our Python files.

You should now have `demo`, `pygodot`, `_demo` and `toolbox` directories in your PyGodot project.


In the `_demo` folder, we’ll start with creating our Python module for the GDNative node we’ll be creating.
We will name it `python_example.py`:
```py
import math
import numpy as np

from godot.bindings.python import nodes
from godot.core.types import Vector2
from godot.core.signal_arguments import SignalArgumentObject, SignalArgumentVector2
from godot.nativescript import register_method, register_property, register_signal


class PythonExample(nodes.Sprite):
    def __init__(self):
        self.time_passed = 0.0
        self.amplitude = 10
        self._position = np.array([0, 0], dtype=np.float32)

    def _process(self, delta):
        self.time_passed += delta

        self._position[0] = self.amplitude + (self.amplitude * math.sin(self.time_passed * 2.0))
        self._position[1] = self.amplitude + (self.amplitude * math.cos(self.time_passed * 1.5))

        self.set_position(self._position)

        self.time_emit += delta

        if self.time_emit >= 2:
            self.emit_signal('position_changed', self, Vector2.from_numpy(self._position))
            self.time_emit = 0

    @staticmethod
    def _register_methods(cls):
        register_method(PythonExample, '_process')
        register_property(PythonExample, 'amplitude', 10)

        register_signal(PythonExample, 'position_changed',
                        SignalArgumentObject('node'), SignalArgumentVector2('new_position'))
```

There is one more Python file we need, it should be named `__init__.py`.  Our GDNative plugin can contain
multiple NativeScripts, each with their own Python module like we’ve implemented `PythonExample` up above.
What we need now is a small bit of code that tells Godot about all the NativeScripts in our GDNative plugin.

```py
from godot.nativescript import register_class
from . import python_example


def _nativescript_init():
    register_class(python_example.PythonExample)
```

### Creating Godot extension nodes

Create `godot_setup.py` in the root directory:
```py
from godot_tools.setup import godot_setup
from godot_tools.setup.libraries import GenericGDNativeLibrary
from godot_tools.setup.extensions import NativeScript


godot_setup(
    godot_project='demo',
    package='_demo',
    # set_development_path=True,
    library=GenericGDNativeLibrary('gdexample.gdnlib', singleton=False),
    extensions=[
        NativeScript('python_example.gdns', class_name='PythonExample', python_sources=['python_example.py'])
    ]
)
```

Now we can execute the setup script and create our GDNative extensions:

```
$ python godot_setup.py install
```


### Using the GDNative module

Time to jump back into Godot. We load up the main scene we created way back in the beginning and
now add a Sprite to our scene:

[picture]

We’re going to assign the Godot logo to this sprite as our texture, disable the `centered` property and drag
our `cython_example.gdns` file onto the `script` property of the sprite:

[picture]

We’re finally ready to run the project:

[picture]
