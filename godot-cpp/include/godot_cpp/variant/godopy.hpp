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
  void *_owner;
  void *_ref_owner;
  int is_singleton;
  void *__godot_class__;
};

extern PyTypeObject GDPy_ObjectType;
extern PyTypeObject GDPy_ExtensionType;

extern PyObject *_get_object_from_owner(void *, const godot::String &);

#endif
