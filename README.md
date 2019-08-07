# PyGodot

Python and Cython bindings to the [Godot game engine](http://godotengine.org/).

## Work in progress

The bindings are a work in progress. Some planned features are missing and the existing APIs can be unstable!

## Features

- Compilation of Cython and Python code to GDNative binaries
- Generic GDNative library for pure Python code without any native code compilation
- Dynamic loading of Python code and dynamic NativeScript extensions
- Ability to write Cython methods that work as fast as native C++ methods
- Two specialized sets (Cython and Python) of automatically generatated bindings to the full Godot API
- [Cython] Access to the complete official C++ API from Cython programming language
- [Cython] Automatic type conversions between `Variant` and `PyObject *`, `String` and `str`, etc.

This project consists of two parts:
- [**Cython API**](CYTHON_INTRO.md)
- [**Pure Python API**](PYTHON_INTRO.md)
