# GodoPy

Different [Python](https://www.python.org/) bindings to the [Godot game engine](http://godotengine.org/)

## Work in progress

The bindings are a work in progress. Some planned features are missing and the existing APIs can be unstable!

Godot 4 features are not yet supported, master branch tracks Godot 3.2
This will change in the near future, master branch will track the Godot's master

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

If you like to support this project consider becoming a patron on [Patreon](https://www.patreon.com/join/godopy)
