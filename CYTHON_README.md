# Cython API

The Cython API is built on top of the C++ bindings to the NativeScript from
the exisitng [godot-cpp](https://github.com/GodotNativeTools/godot-cpp) project.

## Getting started

### Prerequisites

[**Build latest version of Godot**](https://godot.readthedocs.io/en/latest/development/compiling/index.html)

TODO: Describe Python 3 requirement

### Setting up a new project

This introduction is based on the [GDNative C++ example](https://docs.godotengine.org/en/latest/tutorials/plugins/gdnative/gdnative-cpp-example.html) from the offial Godot docs.

The instructions below assume using git for managing your project.

```
$ mkdir gdnative-cython-example
$ cd gdnative-cython-example
$ git clone --recursive https://github.com/ivhilaire/pygodot
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/ivhilaire/pygodot
$ git submodule update --init --recursive
```

TODO: Build instructions for each platform in separate files

[Ubuntu Linux]
```
sudo apt-get install python3-venv python3-dev
sudo apt-get build-dep python3
```

Install PyGodot and set up a development environment (this process will take some time):
```
$ # Mac and Linux:
$ python3 pygodot/internal_python_build.py
$ pygodot/buildenv/bin/python3 -m pip install -U pip
$ pygodot/buildenv/bin/python3 -m pip install git+https://github.com/cython/cython.git@master#egg=Cython
$ pygodot/buildenv/bin/python3 -m pip install -r pygodot/internal-requirements.txt
$ python3 -m venv toolbox
$ source toolbox/bin/activate
(toolbox) $ pip install -r pygodot/godot-tools-requirements.txt
(toolbox) $ export GODOT_BUILD=<path to Godot source folder>
(toolbox) $ pygodot/bootstrap.py
(toolbox) $ cd pygodot
(toolbox) $ scons -j4 only_cython=yes
(toolbox) $ scons -j4
(toolbox) $ cd ..
(toolbox) $ pip install -e pygodot
```
> Replace `<path to Godot source folder>` with an actual path. Godot source should be compiled.
> When you finish working with a virtual environment, run `deactivate` command
> Cython must be installed before numpy because numpy build depends on it

[Windows only] `choco install mingw` for `cpp.exe` (needed by bootstrap script)

[Windows only] If you are using Windows PowerShell, first run as admin: `set-executionpolicy RemoteSigned`
```
> # Windows:
> python .\pygodot\internal_python_build.py
> python .\pygodot\internal_python_build.py target=release
> .\pygodot\deps\python\PCbuild\amd64\python_d.exe -m venv .\pygodot\buildenv
> .\pygodot\buildenv\Scripts\activate
(buildenv) > python_d -m pip install --upgrade pip
(buildenv) > cp .\pygodot\deps\python\PC\pyconfig.h .\pygodot\buildenv\Include\
(buildenv) > python_d -m pip install git+https://github.com/cython/cython.git@master#egg=Cython
(buildenv) > python -m pip install git+https://github.com/numpy/numpy.git@master#egg=numpy
(buildenv) > python_d -m pip install -r .\pygodot\internal-requirements.txt
(buildenv) > deactivate
> python -m venv toolbox
> .\toolbox\Scripts\activate
(toolbox) > python -m pip install -r pygodot\godot-tools-requirements.txt
(toolbox) > $env:GODOT_BUILD = 'C:\path\to\godot'
(toolbox) > cd pygodot
(toolbox) $ python bootstrap.py
(toolbox) $ scons -j4 only_cython=yes
(toolbox) $ scons -j4
(toolbox) $ cd ..
(toolbox) $ pip install -e pygodot
```
> Replace `C:\path\to\godot` with an actual path.
> When you finish working with a virtual environment, run `deactivate` command
> Cython must be installed before numpy because numpy build depends on it
> Debug build of Python couldn't build numpy on Windows therefore instructions use release build


### Creating a GDNative extension

Now it’s time to build an actual extension. We’ll start by creating an empty Godot project
in which we’ll place a few files.

Open Godot and create a new project. For this example, we will place it in a folder called `demo` inside our PyGodot project’s folder structure.

In our demo project, we’ll create a scene containing a Node2D called “Main” and we’ll save it as `main.tscn`.
We’ll come back to that later.

Back in the top-level project folder, we’re also going to create a subfolder called `src`
in which we’ll place our source files.

You should now have `demo`, `pygodot` and `src` directories in your PyGodot project.

Place an empty file `__init__.py` inside the `src` folder:
```
$ touch src/__init__.py
```

In the `src` folder, we’ll start with creating our Cython definition file for the GDNative node we’ll be creating.
We will name it `gdexample.pxd`:
```pyx
from godot.bindings.cython cimport nodes


cdef class GDExample(nodes.Sprite):
    cdef float time_passed

    cdef _process(GDExample self, float delta)
```
> Note the `cdef` declarations and that `cimport` is not the same as `import`

...

Let’s implement our functions by creating our `gdexample.pyx` file:
```pyx
from libc.math cimport cos, sin

from godot.bindings.cython cimport nodes
from godot.cpp.core_types cimport Vector2

from godot.nativescript cimport register_method


cdef class GDExample(nodes.Sprite):
    def __cinit__(self):
        self.time_passed = 0.0

    cdef _process(self, float delta):
        self.time_passed += delta

        cdef Vector2 new_position = Vector2(
            10.0 + (10.0 * sin(self.time_passed * 2.0)), 10.0 + (10.0 * cos(self.time_passed * 1.5)))

        self.set_position(new_position)

    @staticmethod
    def _register_methods():
        register_method(GDExample, '_process', GDExample._process)
```

Note that `Vector2` is a native C++ type.


...

There is one more Cython file we need, we'll name it `gdlibrary.pyx`.  Our GDNative plugin can contain
multiple NativeScripts, each with their own `.pxd` and `.pyx` file like we’ve implemented
`GDExample` up above. What we need now is a small bit of code that tells Godot about all the NativeScripts in our GDNative plugin.

```pyx
from godot.nativescript cimport register_class

from . cimport gdexample


cdef public _pygodot_nativescript_init():
    register_class(gdexample.GDExample)
```

It is possible to register plain C++ NativeScript classes too. Here's an example:
```pyx
from godot.nativescript cimport register_cpp_class
from godot.bindings.cpp cimport nodes

cdef extern from "gdexample.h" namespace "godot" nogil:
    cdef cppclass GDExample(nodes.Sprite)

cdef public _pygodot_nativescript_init():
    register_cpp_class[GDExample]()
```

### Building the extension

Create the `setup.py` file in the root directory:
```py
from setuptools import setup
from godot_tools.setup import GodotProject, get_cmdclass
from godot_tools.setup.extensions import GDNativeLibrary, NativeScript


setup(
    ext_modules=[
        GodotProject('demo', source='src', binary_path='.bin'),
        GDNativeLibrary('gdexample.gdnlib', source='gdlibrary.pyx'),
        NativeScript('gdexample.gdns', class_name='GDExample', sources=['gdexample.pyx'])
    ],
    cmdclass=get_cmdclass()
)
```

Now we can execute the setup script and build our GDNative extensions:
```
$ python setup.py build_ext -i
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


### Properties and signals

Godot properties and signals are supported.
