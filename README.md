# GodoPy

[Python](https://www.python.org) scripting for the [Godot game engine](http://godotengine.org),
[Cython](https://cython.org) and Python bindings to the Godot script API.

## Work in progress

*This branch contains an initial prototype of the project.*

*New version is being written and it won't be compatible with the implementation in this branch.*

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
