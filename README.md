# GodoPy

[Python](https://www.python.org) scripting for the [Godot game engine](http://godotengine.org),
[Cython](https://cython.org) and Python bindings to the Godot script API.

The goal of this project is to extend Godot with [SciPy ecosystem](https://scipy.org)
and to add interactive scripting features to the editor.

GodoPy extends and complements GDScript, there is no goal to replace or
provide an alternative to anything in the core Godot engine.

## Work in progress

The bindings are a work in progress. Some planned features are missing and the existing APIs can be unstable!

Godot 4 features are not yet supported, master branch tracks the 3.2 branch of Godot.
This will change in the near future, master branch will track the Godot's master.

## Features

* Compilation of Cython and Python code to GDNative binaries

* Running dynamic, byte-compiled and AOT-compiled Python code from NativeScript extensions

* Automatically generatated bindings to the full Godot API

* Access to the complete official C++ API from the Cython programming language, full interoperability
  between `godot-cpp` and `GodoPy`; in fact `godot-cpp` types are used by Cython bindings as is

* Automatic type conversions between Godot and Python types

* NumPy array access to all numeric Godot types


## Installation

* [Python introduction](PYTHON_QUICKSTART.md)

* [Development installation](CYTHON_QUICKSTART.md), required for
  [Cython](https://cython.readthedocs.io/en/latest/src/quickstart/overview.html) workflow,
  AOT compilation or advanced Python workflow


## Donate

If you think that this project is useful, please support us on [Patreon](https://www.patreon.com/join/godopy).
