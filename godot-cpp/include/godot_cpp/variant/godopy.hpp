#ifndef GODOT_GODOPY_HPP
#define GODOT_GODOPY_HPP

#include <Python.h>
#include <gdextension_interface.h>

PyTypeObject Vector2_Type;
PyTypeObject Vector3_Type;
PyTypeObject Vector4_Type;
PyTypeObject Color_Type;

struct GDPy_Object {
  PyObject_HEAD
  void *__pyx_vtab;
  void *_owner;
  int is_singleton;
  GDExtensionInstanceBindingCallbacks _binding_callbacks;
  void *__godot_class__;
};

PyTypeObject GDPy_ObjectType;

#endif
