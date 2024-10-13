#ifndef GODOT_GODOPY_HPP
#define GODOT_GODOPY_HPP

#include <Python.h>
#include <gdextension_interface.h>
#include <godot_cpp/variant/string_name.hpp>

PyTypeObject Vector2_Type;
PyTypeObject Vector3_Type;
PyTypeObject Vector4_Type;
PyTypeObject Color_Type;

struct GDPy_Object {
  PyObject_HEAD
  void *__pyx_vtab;
  void *_owner;
  void *_ref_owner;
  int is_singleton;
  GDExtensionInstanceBindingCallbacks _binding_callbacks;
  void *__godot_class__;
};

struct GDPy_Extension {
  struct GDPy_Object __pyx_base;
  godot::StringName _godot_class_name;
  godot::StringName _godot_base_class_name;
  int _needs_cleanup;
};

extern PyTypeObject GDPy_ObjectType;
extern PyTypeObject GDPy_ExtensionType;

#endif
