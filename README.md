# PyGodot

Python and Cython bindings to the [Godot game engine](http://godotengine.org/).

The goal of this project is to provide access to the modern Game Development tools to Python developers and to make the Python development ecosystem, including powerful scientific and machine learning tools, accessible to the Godot community.

There is no goal to replace GDScript with Python or to make a better GDScript. The idea is to empower developers
with a set of high level tools for tasks that could be done only in C++.

## Features

- Compilation of Cython and Python code to GDNative binaries
- Generic GDNative library for pure Python code without any native code compilation
- Dynamic loading of Python code and dynamic NativeScript extensions
- Ability to write Cython methods that work as fast as native C++ methods
- Two specialized sets (Cython and Python) of automatically generatated bindings to the full Godot API
- [Cython] Access to the complete official C++ API from Cython programming language
- [Cython] Automatic type conversions between `Variant` and `PyObject *`, `String` and `str`, etc.

## Work in progress

The bindings are a work in progress. Some planned features are missing and the existing APIs can be unstable!

This project consists of two parts:
- [**Cython API**](CYTHON_INTRO.md)
- [**Pure Python API**](PYTHON_INTRO.md)
