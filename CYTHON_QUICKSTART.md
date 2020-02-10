# Cython API

The Cython API is built on top of the [C++ bindings to NativeScript](https://github.com/GodotNativeTools/godot-cpp).

This tutorial is based on the official [GDNative C++ example](https://docs.godotengine.org/en/latest/tutorials/plugins/gdnative/gdnative-cpp-example.html).

## Getting started

### Prerequisites

[**Build latest version of Godot**](https://godot.readthedocs.io/en/latest/development/compiling/index.html)

TODO: Describe Python 3 requirement

### Setting up a new project

The instructions below assume using git for managing your project.

```
$ mkdir gdnative-cython-example
$ cd gdnative-cython-example
$ git clone --recursive https://github.com/godopy/godopy GodoPy
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/godopy/godopy GodoPy
$ git submodule update --init --recursive
```

Build GodoPy and set up a development environment for your platform:
- [MacOS](BUILD_MACOS.md)
- [Linux](BUILD_LINUX.md)
- [Windows](BUILD_WINDOWS.md)


### Creating a GDNative extension

Now it’s time to build an actual extension. We’ll start by creating an empty Godot project
in which we’ll place a few files.

Open Godot and create a new project. For this example, we will place it in a folder called `demo` inside our GodoPy project’s folder structure.

In our demo project, we’ll create a scene containing a Node2D called “Main” and we’ll save it as `main.tscn`.
We’ll come back to that later.

Back in the top-level project folder, we’re also going to create a subfolder called `_demo`
in which we’ll place our source files.

You should now have `demo`, `godopy`, `_demo` and `toolbox` directories in your GodoPy project.

Place an empty file `__init__.pxd` inside the `_demo` folder, Cython requires this in order to perform C-level imports:
```
$ touch _demo/__init__.pxd
```

In the `_demo` folder, we’ll start with creating our Cython definition file for the GDNative node we’ll be creating.
We will name it `cython_example.pxd`:
```pyx
from godot.bindings.cython cimport nodes


cdef class CythonExample(nodes.Sprite):
    cdef float time_passed

    cdef _process(CythonExample self, float delta)
```
> Note the `cdef` declarations and that `cimport` is a C-level import, not Python `import`

...

Let’s implement our functions by creating our `cython_example.pyx` file:
```pyx
from libc.math cimport cos, sin

from godot.bindings.cython cimport nodes
from godot.core.cpp_types cimport Vector2

from godot.nativescript cimport register_method


cdef class CythonExample(nodes.Sprite):
    def __cinit__(self):
        self.time_passed = 0.0

    cdef _process(self, float delta):
        self.time_passed += delta

        cdef Vector2 new_position = Vector2(
            10.0 + (10.0 * sin(self.time_passed * 2.0)), 10.0 + (10.0 * cos(self.time_passed * 1.5)))

        self.set_position(new_position)

    @staticmethod
    def _register_methods():
        register_method(CythonExample, '_process', CythonExample._process)
```

Note that `Vector2` is a native C++ type and `register_method` is a C++ template that wraps
low level C function pointers. This completely eliminates any Python runtime overhead.

...

There is one more Cython file we need, `__init__.pyx`.  Our GDNative plugin can contain
multiple NativeScripts implemented in C++, Cython or pure Python, each with their own files like we’ve implemented `CythonExample` up above. What we need now is a small bit of code that tells Godot about all the NativeScripts in our GDNative plugin.

```cython
from godot.nativescript cimport register_class

from . cimport cython_example


cdef public _godopy_nativescript_init():
    register_class(cython_example.CythonExample)
```

It is possible to register plain C++ NativeScript classes too. Here's an example:
```cython
from godot.nativescript cimport register_cpp_class
from godot.bindings.cpp cimport nodes

cdef extern from "gdexample.h" namespace "godot" nogil:
    cdef cppclass GDExample(nodes.Sprite)

cdef public _godopy_nativescript_init():
    register_cpp_class[GDExample]()
```
> This will register a C++ class from the [C++ tutorial](https://docs.godotengine.org/en/latest/tutorials/plugins/gdnative/gdnative-cpp-example.html)

Also, is is possible to mix pure Python classes with Cython classes:
```cython
from godot.nativescript cimport register_class
from godot.bindings.cpp cimport nodes

from godot.utils cimport allow_pure_python_imports

from _demo cimport cython_example

# sys.modules['_demo'] is configured to work with Cython-compiled modules
# by default and won't be able to import our pure Python files without the "allow_pure_python_imports"
allow_pure_python_imports('_demo')
from _demo import python_example

cdef public _godopy_nativescript_init():
    register_class(cython_example.CythonExample)
    register_class(python_example.PythonExample)
```

### Building the extension

Create a `project.py` file in the `_demo` directory:
```py
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parents[1]

GODOT_PROJECT = 'demo'
PYTHON_PACKAGE = '_demo'
GDNATIVE_LIBRARY = 'cython_example.gdnlib'

NATIVESCRIPT_EXTENSIONS = [
    'CythonExample'
]

NATIVESCRIPT_SOURCES = {
    'CythonExample': ['cython_example.pyx']
}
```

Now we can execute the setup script and build our GDNative extensions:
```
$ GODOPY_PROJECT_MODULE=_demo godopy installscripts
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


### Properties and signals

Godot properties and signals are supported.
