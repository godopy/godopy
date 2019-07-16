# pygodot
Python and Cython bindings for the Godot script APIs

The goal of this project is to provide Python and Cython language support for the Godot extension development.

This project is built on top of [godot-cpp](https://github.com/GodotNativeTools/godot-cpp).

Index:
-   [**Getting Started**](#getting-started)
-   [**Building Native Godot Extensions**](#building-native-godot-extensions)

## Getting Started

### Setting up a new project

```
$ mkdir SimpleProject
$ cd SimpleProject
$ python3 -m venv env
$ source env/bin/activate
$ git clone https://github.com/ivhilaire/pygodot # or unpack .zip from github
(env) $ pip install -e ./pygodot
```

### Creating a simple class

Create `simple.py` and add the following code
```py
from godot import nodes, gdnative, print

class SimpleClass(nodes.Reference):
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
from godot import gdnative

def nativescript_init():
    gdnative.register_class(simple.SimpleClass)
```

### Creating .gdnlib and .gdns files

*TODO:* Create specialized `pygodot <cmd>` commands
```
(env) $ cd pygodot
(env) # python setup.py py2app
(env) # cd ..
(env) $ mkdir demo
(env) $ touch demo/project.godot
(env) $ mkdir demo/bin # TODO: automate gdnlib setup
(env) $ cd demo/bin
(env) $ ln -s ../../pygodot/bin/lib* .
(env) $ ln -s ../../pygodot/dist/pygodot.app/Contents/Resources pyres
(env) $ cd ..
```

Create `pygodot.gdns` under `demo/bin` and add the following code
```toml
[general]

singleton=false
load_once=true
symbol_prefix="godot_"
reloadable=false

[entry]

X11.64="res://bin/libpygodot.linux.debug.64.so"
Windows.64="res://bin/libpygodot.windows.debug.64.dll"
OSX.64="res://bin/libpygodot.osx.debug.64.dylib"

[dependencies]

X11.64=[]
Windows.64=[]
OSX.64=[]
```

Open Godot editor
```
(env) $ godot -e
```

[Describe .gdns resource creation]

### Implementing with gdscript
```gdscript
var simpleclass = load("res://bin/simpleclass.gdns").new();
simpleclass.test_method();
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
(env) $ pip install scons
(env) $ cd pygodot
(env) $ cp -R <path to godot build>/modules/gdnative/include ./godot_headers
(env) $ godot --gdnative-generate-json-api godot_headers/api.json
(env) $ pygodot genapi
```

### Compiling libraries

```
(env) $ cython -3 --cplus -o godot/Godot.cpp godot/Godot.pyx # TODO: automate
(env) $ cython -3 --cplus -o godot/Bindings.cpp godot/Bindings.pyx # TODO: automate
(env) $ scons platform=<your platform> generate_bindings=yes
```

> Replace `<your platform>` with either `windows`, `linux` or `osx`.

> [Add other notes]

> The resulting libraries will be created in `pygodot/bin/`, take note of their names as they will be different depending on platform.

### To be consintued…

[Describe steps to build native extensions]
