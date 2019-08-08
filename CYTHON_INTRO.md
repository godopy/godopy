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
$ git clone --recursive https://github.com/ivhilaire/pygodot
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/ivhilaire/pygodot
$ git submodule update --init --recursive
```

[Ubuntu Linux] `sudo apt-get build-dep python3.7`

Install PyGodot and set up a development environment (this process will take some time):
```
$ # Mac and Linux:
$ python3 pygodot/build_python.py
$ pygodot/buildenv/bin/python3 -m venv _meta
$ source _meta/bin/activate
(_meta) $ python -m pip install -r pygodot/meta-requirements.txt
(_meta) $ export GODOT_BUILD=<path to Godot source folder>
(_meta) $ cd pygodot
(_meta) $ python bootstrap.py
(_meta) $ python setup.py develop
```
> Replace `<path to Godot source folder>` with an actual path. Godot source should be compiled.
> When you finish working with a virtual environment, run `deactivate` command

[Windows only] If you are using Windows PowerShell, first run as admin: `set-executionpolicy RemoteSigned`
```
$ # Windows:
$ python pygodot/build_python.py
$ .\pygodot\deps\python\pcbuild\amd64\py -m venv _meta
$ .\_meta\Scripts\Activate
(_meta) $ python -m pip install -r pygodot/meta-requirements.txt
(_meta) $ $env:GODOT_BUILD = 'C:\path\to\godot'
(_meta) $ cd pygodot
(_meta) $ python bootstrap.py
(_meta) $ python setup.py develop
```
> Replace `C:\path\to\godot` with an actual path.
> When you finish working with a virtual environment, run `deactivate` command


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
from pygodot.build import GodotProject, get_cmdclass
from pygodot.build.extensions import GDNativeLibrary, NativeScript


setup(
    name='demo',
    version='0.0.1',
    packages=['demo'],
    ext_modules=[
        GodotProject('demo', source='src', binary_path='.bin'),
        GDNativeLibrary('gdlibrary.gdnlib', source='gdlibrary.pyx'),
        NativeScript('gdexample.gdns', class_name='GDExample', sources=['gdexample.pyx'])
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


### Properties and signals

Godot properties and signals are supported.
