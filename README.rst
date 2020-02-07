======
GodoPy
======

Different Python bindings to the Godot game engine.


Work in progress
================

The bindings are a work in progress. Some planned features are missing and the existing APIs can be unstable!


Features
========

* Compilation of Cython and Python code to GDNative binaries

* Running dynamic, byte-compiled and AOT-compiled Python code from NativeScript extensions

* Automatically generatated bindings to the full Godot API

* Access to the complete official C++ API from the Cython programming language, full interoperability
  between ``godot-cpp`` and ``GodoPy``; in fact ``godot-cpp`` types are used by Cython bindings as is

* Automatic type conversions between Godot and Python types

* NumPy array access to all numeric Godot types


Installation
============

* ``PYTHON_QUICKSTART.md`` introduces GodoPy for beginners

* Advanced usage and AOT compilation are described in ``CYTHON_QUICKSTART.md``
