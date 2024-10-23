#ifndef GODOT_GODOPY_HPP
#define GODOT_GODOPY_HPP

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#endif

#include <Python.h>
#include <gdextension_interface.h>
#include <numpy/arrayobject.h>

#include <godot_cpp/variant/variant.hpp>

extern PyObject *variant_to_pyobject(godot::Variant const &);
extern void variant_from_pyobject(PyObject *, godot::Variant *);

#endif
