#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "CoreTypes.hpp"
#include "godot/core_types.h"

namespace godot {

PyObject *AABB::pythonize() { return _aabb_to_python(*this); }
PyObject *Array::pythonize() { return _array_to_python(*this); }
PyObject *Color::pythonize() { return _color_to_python(*this); }
PyObject *Dictionary::pythonize() { return _dictionary_to_python(*this); }
PyObject *NodePath::pythonize() { return _nodepath_to_python(*this); }
PyObject *Plane::pythonize() { return _plane_to_python(*this); }
PyObject *PoolByteArray::pythonize() { return _poolbytearray_to_python(*this); }
PyObject *PoolIntArray::pythonize() { return _poolintarray_to_python(*this); }
PyObject *PoolRealArray::pythonize() { return _poolrealarray_to_python(*this); }
PyObject *PoolStringArray::pythonize() { return _poolstringarray_to_python(*this); }
PyObject *PoolVector2Array::pythonize() { return _poolvector2array_to_python(*this); }
PyObject *PoolVector3Array::pythonize() { return _poolvector3array_to_python(*this); }
PyObject *PoolColorArray::pythonize() { return _poolcolorarray_to_python(*this); }
PyObject *Quat::pythonize() { return _quat_to_python(*this); }
PyObject *Rect2::pythonize() { return _rect2_to_python(*this); }
PyObject *RID::pythonize() { return _rid_to_python(*this); }
PyObject *CharString::pythonize() { return _charstring_to_python(*this); }
PyObject *String::pythonize() { return _string_to_python(*this); }
PyObject *Transform::pythonize() { return _transform_to_python(*this); }
PyObject * Transform2D::pythonize() { return _transform2d_to_python(*this); }
PyObject *Vector2::pythonize() { return _vector2_to_python(*this); }
PyObject *Vector3::pythonize() { return _vector3_to_python(*this); }

} // namespace godot
