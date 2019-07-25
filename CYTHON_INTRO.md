# Cython API

The Cython API is built on top of the C++ bindings to the NativeScript from
the exisitng [godot-cpp](https://github.com/GodotNativeTools/godot-cpp) project.

## Getting started

### Prerequisites

[**Build latest version of Godot**](https://godot.readthedocs.io/en/latest/development/compiling/index.html)

[**Make sure you’ve got Python & pip**](https://docs.python-guide.org/dev/virtualenvs/#make-sure-you-ve-got-python-pip)

### Setting up a new project

This introduction is based on the [GDNative C++ example](https://docs.godotengine.org/en/latest/tutorials/plugins/gdnative/gdnative-cpp-example.html) from the offial Godot docs.

The instructions below assume using git for managing your project.

```
$ mkdir gdnative-cython-example
$ cd gdnative-cython-example
$ git clone https://github.com/ivhilaire/pygodot PyGodot
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/ivhilaire/pygodot PyGodot
$ git submodule update --init --recursive
```

Install PyGodot and create the development environment
```
$ pipenv install -e PyGodot
```

### Copying the development headers and building the bindings

```
$ cp -R <path to the Godot source folder>/modules/gdnative/include PyGodot/godot_headers
$ godot --gdnative-generate-json-api PyGodot/godot_headers/api.json
$ pipenv run pygodot genapi
$ pipenv run pygodot genbindings
$ pipenv shell
(SimpleProject) $ cd PyGodot
(SimpleProject) $ python setup.py develop --generate_bindings
(SimpleProject) $ exit
```
> Replace `<path to the Godot source folder>` with an actual path. Godot source should be compiled.


### Creating a GDNative extension

Now it’s time to build an actual extension. We’ll start by creating an empty Godot project
in which we’ll place a few files.

Open Godot and create a new project. For this example, we will place it in a folder called `demo` inside our PyGodot project’s folder structure.

In our demo project, we’ll create a scene containing a Node2D called “Main” and we’ll save it as `main.tscn`.
We’ll come back to that later.

Back in the top-level project folder, we’re also going to create a subfolder called `_demo`
in which we’ll place our source files.

You should now have `demo`, `PyGodot` and `_demo` directories in your PyGodot project.

Place an empty file `__init__.py` inside the `_demo` folder:
```
$ touch _demo/__init__.py
```

This will turn our `_demo` folder into a Python [package](https://docs.python.org/3/glossary.html#term-regular-package).

In the `_demo` folder, we’ll start with creating our Cython declaration file for the GDNative node we’ll be creating.
We will name it `gdexample.pxd`:
```pyx
from pygodot.cnodes cimport Sprite


cdef class GDExample(Sprite):
    cdef float time_passed

    cpdef _process(GDExample self, float delta)
```
> Note the `cdef` declarations and that `cimport` is not the same as `import`

...

Let’s implement our functions by creating our `gdexample.pyx` file::
```pyx
from libc.math cimport cos, sin
from pygodot.cnodes cimport Sprite
from pygodot.cctypes cimport Vector2

from pygodot.gdnative cimport register_method

cdef class GDExample(Sprite):
    def __cinit__(self):
        self.time_passed = 0.0

    cpdef _process(self, float delta):
        self.time_passed += delta

        cdef Vector2 new_position = Vector2(
            10.0 + (10.0 * sin(self.time_passed * 2.0)), 10.0 + (10.0 * cos(self.time_passed * 1.5)))

        self.set_position(new_position)

    @classmethod
    def _register_methods(cls):
        register_method(cls, cls._process)
```

Note that `Vector2` is a native C++ type.


...

There is one more Cython file we need, we'll name it `gdlibrary.pyx`.  Our GDNative plugin can contain
multiple NativeScripts, each with their own `.pxd` and `.pyx` file like we’ve implemented
`GDExample` up above. What we need now is a small bit of code that tells Godot about all the NativeScripts in our GDNative plugin.

```pyx
from godot_headers.gdnative_api cimport *

from pygodot.gdnative cimport register_class
from .gdexample cimport GDExample

cdef public int _pygodot_nativescript_init(godot_gdnative_init_options options) except -1:
    register_class(GDExample)

    return GODOT_OK
```

### Building the extension

Create the `setup.py` file in the root directory:
```py
from setuptools import setup
from pygodot.build import GodotProject, get_cmdclass
from pygodot.build.extensions import GDNativeLibrary, NativeScript


setup(
    name='demo',
    version='0.0.1',
    packages=['_demo'],
    ext_modules=[
        GodotProject('demo', shadow='_demo', binary_path='.bin'),
        GDNativeLibrary('bin/gdexample.gdnlib', source='gdlibrary.pyx'),
        NativeScript('gdexample.gdns', classname='GDExample', sources=['gdexample.pyx'])
    ],
    cmdclass=get_cmdclass()
)
```

Now we can execute the setup script and build our GDNative extensions:
```
$ pipenv shell
$ python setup.py develop
$ exit
```

### Using the GDNative module

Time to jump back into Godot. We load up the main scene we created way back in the beginning and
now add a Sprite to our scene:

[picture]

We’re going to assign the Godot logo to this sprite as our texture, disable the `centered` property and drag
our `gdexample.gdns` file onto the `script` property of the sprite:

[picture]

We’re finally ready to run the project:

[picture]
