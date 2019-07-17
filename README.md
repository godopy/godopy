# PyGodot

Python and Cython bindings [Godot game engine](http://godotengine.org/).

The goal of this project is to provide Python and Cython language support for the Godot extension development.

## Work in progress

The bindings are a work in progress. A lot of planned features are missing and the existing APIs are very unstable!

## Differences from godot-python

Unlike [Godot Python](https://github.com/touilleMan/godot-python), this project focuses on the ability to compile
your Godot modules to the native code and enables lower level access to the Godot C/C++ APIs.

The technical side is that PyGodot is built on top of the NativeScript 1.1 API and
the exisitng [godot-cpp](https://github.com/GodotNativeTools/godot-cpp) bindings, Godot Python does not provide
access to these APIs.

And, finally, PyGodot tries to integrate into the existing Python ecosystem and play by its rules: it works with
PIP, virtual environments and allows to interact with external Python dependencies.

Index:
-   [**Getting Started**](#getting-started)
-   [**Building Native Godot Extensions**](#building-native-godot-extensions)

## Getting Started

### Setting up a new project

```
$ mkdir SimpleProject
$ cd SimpleProject
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ git clone https://github.com/ivhilaire/pygodot
(venv) $ pip install -e ./pygodot
```

### Creating a simple class

Create `simple.py` and add the following code
```py
from godot import nodes, gdnative, print

class Simple(nodes.Reference):
    def test_method(self):
        print('This is test!')

    def method(self, arg):
        return arg

    @classmethod
    def _register_methods(cls):
        gdnative.register_method(cls, 'method')
        gdnative.register_method(cls, 'test_method')
```

Provide an entry point for NativeScript, create `gdlibrary.py`:
```py
import simple
from godot import gdnative

def nativescript_init():
    gdnative.register_class(simple.Simple)
```

### Initializing Python environment inside the Godot project

There is one more file we need, create `setup.py`:

```py
from setuptools import setup

setup(
    app=['simple.py'],
    data_files=[],
    options={'py2app': {}},
    setup_requires=['py2app'],
)
```

> This will work only on a Mac, support for Windows and Linux is coming!

Create a new Godot project. For this example we will place it in a folder called `demo` inside `SimpleProject`.

Generate a Python environment for your scripts and link it to the Godot project

```
(venv) $ python setup.py py2app
(venv) $ mkdir demo/bin
(venv) $ cd demo/bin
(venv) $ ln -s ../../dist/simple.app/Contents/Resources pygodot.resources
(venv) $ cd ../..
```

### Installing Godot resource files

Create a new Godot project. For this example we will place it in a folder called `demo` inside `SimpleProject`.

Install your script as a NativeScript resource:

```
(venv) $ pygodot install demo/bin
(venv) $ pygodot installscript demo/bin Simple
(venv) $ cd demo/bin
(venv) $ ln -s ../../pygodot/dist/pygodot.app/Contents/Resources pygodot.resources
(venv) $ cd ..
(venv) $ godot -e
```

Your Python script is now ready to use from Godot, it is called `simple.gdns` inside `bin` folder.

...

### Implementing with gdscript
```gdscript
var simple = load("res://bin/simple.gdns").new()
simple.test_method()
```

## Building Native Godot Extensions

| **Build latest version of Godot** | [**GitHub**](https://github.com/godotengine/godot) | [**Docs**](https://godot.readthedocs.io/en/latest/development/compiling/index.html) |
| --- | --- | --- |

### Setting up a new project

The instructions below assume using git for managing your project.

```
$ mkdir SimpleLibrary
$ cd SimpleLibrary
$ git clone https://github.com/ivhilaire/pygodot
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/ivhilaire/pygodot
$ git submodule update --init --recursive
```

Initialize the environment and install the dependencies
```
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install -e ./pygodot  # "develop" install
```

### Updating the Godot development headers

```
(venv) $ cd pygodot
(venv) $ cp -R <path to godot build>/modules/gdnative/include godot/headers
(venv) $ godot --gdnative-generate-json-api godot/headers/api.json
(venv) $ pygodot genapi
```

### Compiling libraries

```
(venv) $ python setup.py develop --generate_bindings 
```

> The resulting libraries will be created in `pygodot/`, take note of their names as they will be different depending on platform.

### To be consintued…

[Describe steps to build native extensions]
