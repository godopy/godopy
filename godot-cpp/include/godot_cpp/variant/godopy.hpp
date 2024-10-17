#ifndef GODOT_GODOPY_HPP
#define GODOT_GODOPY_HPP

#include <Python.h>
#include <gdextension_interface.h>
#include <godot_cpp/variant/string_name.hpp>

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
