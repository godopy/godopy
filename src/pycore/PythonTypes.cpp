#ifndef NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#include "PythonGlobal.hpp"

#include "CoreTypes.hpp"
#include <_lib/godot/core/types.hpp>

namespace godot {

PyObject *AABB::py_wrap() const { return _aabb_to_python_wrapper(*this); }
PyObject *Array::py_wrap() const { return _godot_array_to_python_wrapper(*this); }
PyObject *Basis::py_wrap() const { return _basis_to_python_wrapper(*this); }

PyObject *Color::py_wrap() const { return _color_to_python_wrapper(*this); }
PyObject *Color::py_ndarray() const { return _color_to_numpy(*this); }

PyObject *Dictionary::py_wrap() const { return _godot_dictionary_to_python_wrapper(*this); }
PyObject *NodePath::py_wrap() const { return _nodepath_to_python_wrapper(*this); }
PyObject *Plane::py_wrap() const { return _plane_to_python_wrapper(*this); }

PyObject *PoolByteArray::py_wrap() const { return _poolbytearray_to_python_wrapper(*this); }
PyObject *PoolByteArray::py_read() const { return _poolbytearray_to_python_read(*this); }
PyObject *PoolByteArray::py_write() const { return _poolbytearray_to_python_write(*this); }
PyObject *PoolByteArray::py_ndarray(bool writable) const { return writable ? _poolbytearray_to_numpy(*this) : _poolbytearray_to_numpy_ro(*this); }

PyObject *PoolIntArray::py_wrap() const { return _poolintarray_to_python_wrapper(*this); }
PyObject *PoolIntArray::py_read() const { return _poolintarray_to_python_read(*this); }
PyObject *PoolIntArray::py_write() const { return _poolintarray_to_python_write(*this); }
PyObject *PoolIntArray::py_ndarray(bool writable) const { return writable ? _poolintarray_to_numpy(*this) : _poolintarray_to_numpy_ro(*this); }

PyObject *PoolRealArray::py_wrap() const { return _poolrealarray_to_python_wrapper(*this); }
PyObject *PoolRealArray::py_read() const { return _poolrealarray_to_python_read(*this); }
PyObject *PoolRealArray::py_write() const { return _poolrealarray_to_python_write(*this); }
PyObject *PoolRealArray::py_ndarray(bool writable) const { return writable ? _poolrealarray_to_numpy(*this) : _poolrealarray_to_numpy_ro(*this); }

PyObject *PoolStringArray::py_wrap() const { return _poolstringarray_to_python_wrapper(*this); }
PyObject *PoolStringArray::py_read() const { return _poolstringarray_to_python_read(*this); }
PyObject *PoolStringArray::py_write() const { return _poolstringarray_to_python_write(*this); }
PyObject *PoolStringArray::py_ndarray(bool writable) const { return writable ? _poolstringarray_to_numpy(*this) : _poolstringarray_to_numpy_ro(*this); }

PyObject *PoolVector2Array::py_wrap() const { return _poolvector2array_to_python_wrapper(*this); }
PyObject *PoolVector2Array::py_read() const { return _poolvector2array_to_python_read(*this); }
PyObject *PoolVector2Array::py_write() const { return _poolvector2array_to_python_write(*this); }
PyObject *PoolVector2Array::py_ndarray(bool writable) const { return writable ? _poolvector2array_to_numpy(*this) : _poolvector2array_to_numpy_ro(*this); }

PyObject *PoolVector3Array::py_wrap() const { return _poolvector3array_to_python_wrapper(*this); }
PyObject *PoolVector3Array::py_read() const { return _poolvector3array_to_python_read(*this); }
PyObject *PoolVector3Array::py_write() const { return _poolvector3array_to_python_write(*this); }
PyObject *PoolVector3Array::py_ndarray(bool writable) const { return writable ? _poolvector3array_to_numpy(*this) : _poolvector3array_to_numpy_ro(*this); }

PyObject *PoolColorArray::py_wrap() const { return _poolcolorarray_to_python_wrapper(*this); }
PyObject *PoolColorArray::py_read() const { return _poolcolorarray_to_python_read(*this); }
PyObject *PoolColorArray::py_write() const { return _poolcolorarray_to_python_write(*this); }
PyObject *PoolColorArray::py_ndarray(bool writable) const { return writable ? _poolcolorarray_to_numpy(*this) : _poolcolorarray_to_numpy_ro(*this); }

PyObject *Quat::py_wrap() const { return _quat_to_python_wrapper(*this); }
PyObject *Rect2::py_wrap() const { return _rect2_to_python_wrapper(*this); }
PyObject *RID::py_wrap() const { return _rid_to_python_wrapper(*this); }
PyObject *String::py_wrap() const { return _godot_string_to_python_wrapper(*this); }
PyObject *Transform::py_wrap() const { return _transform_to_python_wrapper(*this); }
PyObject *Transform2D::py_wrap() const { return _transform2d_to_python_wrapper(*this); }

PyObject *Vector2::py_wrap() const { return _vector2_to_python_wrapper(*this); }
PyObject *Vector2::py_ndarray() const { return _vector2_to_numpy(*this); }

PyObject *Vector3::py_wrap() const { return _vector3_to_python_wrapper(*this); }
PyObject *Vector3::py_ndarray() const { return _vector3_to_numpy(*this); }


Vector2 Vector2_from_PyObject(PyObject *obj) {
  if (Py_TYPE(obj) == PyGodotWrapperType_Vector2) {
    return *(Vector2 *)_python_wrapper_to_vector2(obj);
  }

  if (PyArray_Check(obj)) {
    return Vector2((PyArrayObject *)obj);
  }

  throw std::invalid_argument("incompatible Python object argument");
}

Vector3 Vector3_from_PyObject(PyObject *obj) {
  if (Py_TYPE(obj) == PyGodotWrapperType_Vector3) {
    return *(Vector3 *)_python_wrapper_to_vector3(obj);
  }

  if (PyArray_Check(obj)) {
    return Vector3((PyArrayObject *)obj);
  }

  throw std::invalid_argument("incompatible Python object argument");
}

Color Color_from_PyObject(PyObject *obj) {
  if (Py_TYPE(obj) == PyGodotWrapperType_Color) {
    return *(Color *)_python_wrapper_to_color(obj);
  }

  if (PyArray_Check(obj)) {
    return Color((PyArrayObject *)obj);
  }

  throw std::invalid_argument("incompatible Python object argument");
}

} // namespace godot
