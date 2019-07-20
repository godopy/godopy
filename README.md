# PyGodot

Python and Cython bindings for [Godot game engine](http://godotengine.org/).

## Work in progress

The bindings are a work in progress. A lot of planned features are missing and the existing APIs are unstable!

PyGodot is built on top of the C++ bindings to the NativeScript from
the exisitng [godot-cpp](https://github.com/GodotNativeTools/godot-cpp) project.

The Cython API is based on NativeScript 1.1 and the higher-level Python API is built
on top of the Cython API and PluginScript.

Index:
-   [**Getting Started**](#getting-started)
-   [**Building Native Godot Extensions**](#building-native-godot-extensions)

## Getting Started

### Setting up a new project

The instructions below assume using git for managing your project.

| **Build latest version of Godot** | [**GitHub**](https://github.com/godotengine/godot) | [**Docs**](https://godot.readthedocs.io/en/latest/development/compiling/index.html) |
| --- | --- | --- |

```
$ mkdir SimpleProject
$ cd SimpleProject
$ git clone https://github.com/ivhilaire/pygodot
```

If your project is an existing repository, use git submodule instead:
```
$ git submodule add https://github.com/ivhilaire/pygodot
$ git submodule update --init --recursive
```

Install PyGodot and create the development environment (TODO: add links with the detailed instructions on python setup)
```
$ pipenv install -e pygodot
```

### Copying the development headers and generating binding

```
$ cp -R <path to godot build>/modules/gdnative/include pygodot/pygodot/headers
$ godot --gdnative-generate-json-api pygodot/pygodot/headers/api.json
$ pipenv run pygodot genapi
$ pipenv run pygodot genbindings
$ pipenv shell
(SimpleProject) $ cd pygodot
(SimpleProject) $ python setup.py develop --generate_bindings
(SimpleProject) $ exit
```

### Creating a simple class

Create `simple.py` and add the following code
```py
from pygodot import nodes, gdnative, print

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

There is one more file we need, create `gdlibrary.py`:
```py
import simple
from pygodot import gdnative

def nativescript_init():
    gdnative.register_class(simple.Simple)
```

### Installing Godot resource files

Create a new Godot project. For this example we will place it in a folder called `demo` inside `SimpleProject`.

Install your script as a NativeScript resource:

```
(venv) $ pygodot install demo/bin
(venv) $ pygodot installscript demo/bin Simple
(venv) $ godot --path demo -e
```

Your Python script is now ready to use in Godot, it is called `simple.gdns` inside the `bin` folder.

...

### Implementing with gdscript
```gdscript
var simple = load("res://bin/simple.gdns").new()
simple.test_method()
```

## Building Native Godot Extensions

...

## FAQ

### Differences from godot-python

Unlike [Godot Python](https://github.com/touilleMan/godot-python), this project focuses on the ability to compile
your Godot modules to the native code and enables lower level access to the Godot C/C++ APIs.
