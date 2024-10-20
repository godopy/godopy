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


// Old simple PyStructSequences, will be removed
extern PyTypeObject Vector2_Type;
extern PyTypeObject Vector2i_Type;
extern PyTypeObject Size2_Type;
extern PyTypeObject Rect2_Type;
extern PyTypeObject Rect2i_Type;
extern PyTypeObject Vector3_Type;
extern PyTypeObject Vector3i_Type;
extern PyTypeObject Vector4_Type;
extern PyTypeObject Vector4i_Type;
extern PyTypeObject Plane_Type;
extern PyTypeObject Quaternion_Type;
extern PyTypeObject AABB_Type;
extern PyTypeObject Transform3D_Type;
extern PyTypeObject Color_Type;


// gdextension.Object instance
struct GDPyObject {
  PyObject_HEAD
  void *_owner;
  void *_ref_owner;
  int is_singleton;
  void *__godot_class__;
};

// gdextension.Object type
extern PyTypeObject GDPyObject_Type;

// Type conversion functions
extern PyObject *object_to_pyobject(void *);
extern PyObject *variant_object_to_pyobject(godot::Variant const &);
extern void object_from_pyobject(PyObject *, void **);
extern void variant_object_from_pyobject(PyObject *, godot::Variant *);

extern PyObject *bool_to_pyobject(GDExtensionBool);
extern PyObject *variant_bool_to_pyobject(godot::Variant const &);
extern void bool_from_pyobject(PyObject *, GDExtensionBool *);
extern void *variant_bool_from_pyobject(PyObject *, godot::Variant *);
extern PyObject *int_to_pyobject(int64_t);
extern PyObject *variant_int_to_pyobject(godot::Variant const &);
extern void int_from_pyobject(PyObject *, int64_t *);
extern void variant_int_from_pyobject(PyObject *, godot::Variant *);
extern PyObject *float_to_pyobject(double);
extern PyObject *variant_float_to_pyobject(godot::Variant const &);
extern void float_from_pyobject(PyObject *, double *);
extern void variant_float_from_pyobject(PyObject *, godot::Variant *);
extern PyObject *string_to_pyobject(godot::String const &);
extern PyObject *variant_string_to_pyobject(godot::Variant const &);
extern void string_from_pyobject(PyObject *, godot::String *);
extern void variant_string_from_pyobject(PyObject *, godot::Variant *);
extern PyObject *vector2_to_pyobject(godot::Vector2 &);
extern PyObject *vector2i_to_pyobject(godot::Vector2i &);
extern PyObject *variant_vector2_to_pyobject(godot::Variant const &);
extern PyObject *variant_vector2i_to_pyobject(godot::Variant const &);
extern void vector2_from_pyobject(PyObject *, godot::Vector2 *);
extern void vector2i_from_pyobject(PyObject *, godot::Vector2i *);
extern void variant_vector2_from_pyobject(PyObject *, godot::Variant *);
extern void variant_vector2i_from_pyobject(PyObject *, godot::Variant *);

#endif
