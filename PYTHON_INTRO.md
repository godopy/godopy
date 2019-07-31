# Pure Python API

**DOES NOT WORK YET**

The pure Python API is differs from the Cython API in the following ways:
* Compilation is not required
* Python wrappers should be used in place of native C++ classes
* GDNative library file must be called exactly `gdlibrary.py` and located above the Godot project directory

## Getting started

Please follow the instructions from [Cython intro](/CYTHON_INTRO.md#getting-started) to build PyGodot.

Simple `pip install pygodot` setup will be available the future versions.

### Creating pure Python GDNative extension

We’ll start by creating an empty Godot project in which we’ll place a few files.

Open Godot and create a new project. For this example, we will place it in a folder called `demo` inside our PyGodot project’s folder structure.

In our demo project, we’ll create a scene containing a Node2D called “Main” and we’ll save it as `main.tscn`.
We’ll come back to that later.

Back in the top-level project folder, we’re also going to create a subfolder called `_demo`
in which we’ll place our Python files.

You should now have `demo`, `PyGodot` and `_demo` directories in your PyGodot project.

Place an empty file `__init__.py` inside the `_demo` folder:
```
$ touch _demo/__init__.py
```

This will turn our `_demo` folder into a Python [package](https://docs.python.org/3/glossary.html#term-regular-package).

In the `_demo` folder, we’ll start with creating our Python module for the GDNative node we’ll be creating.
We will name it `gdexample.py`:
```py
from pygodot import gdnative, nodes
from math import sin, cos


class GDExample(nodes.Sprite):
    def __init__(self):
        self.time_passed = 0.0

    def _process(self, delta):
        self.time_passed += delta

        new_position = (10.0 + (10.0 * sin(self.time_passed * 2.0)),
                        10.0 + (10.0 * cos(self.time_passed * 1.5)))

        self.set_position(new_position)

    @classmethod
    def _register_methods(cls):
        gdnative.register_method(cls, cls._process)
```

There is one more Python source file we need, it should be named `gdlibrary.py` and placed in the top level directory,
just above the Godot project.  Our GDNative plugin can contain multiple NativeScripts, each with their
own Python module like we’ve implemented `GDExample` up above. What we need now is a small bit of code
that tells Godot about all the NativeScripts in our GDNative plugin.

```py
from pygodot.gdnative import register_class
from _demo.gdexample import GDExample


def nativescript_init():
    register_class(GDExample)
```

### Creating Godot extension nodes

Create `setup.py` in the root directory:
```py
from setuptools import setup
from pygodot.build import GodotProject, get_cmdclass
from pygodot.build.extensions import GenericGDNativeLibrary, NativeScript


setup(
    name='demo',
    version='0.0.1',
    packages=['_demo'],
    ext_modules=[
        GodotProject('demo', shadow='_demo', binary_path='.bin'),
        GenericGDNativeLibrary('bin/_gdexample.gdnlib'),
        NativeScript('gdexample.gdns', classname='GDExample')
    ],
    cmdclass=get_cmdclass()
)
```

Now we can execute the setup script and create our GDNative extensions:

```
$ pipenv shell
$ python setup.py develop
$ exit
```
