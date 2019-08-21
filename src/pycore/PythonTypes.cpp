#include "PythonGlobal.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "CoreTypes.hpp"
#include <internal-packages/godot/core/types.hpp>

namespace godot {

PyObject *AABB::to_python_wrapper() { return _aabb_to_python_wrapper(*this); }
PyObject *Array::to_python_wrapper() { return _godot_array_to_python_wrapper(*this); }
PyObject *Basis::to_python_wrapper() { return _basis_to_python_wrapper(*this); }
PyObject *Color::to_python_wrapper() { return _color_to_python_wrapper(*this); }
PyObject *Dictionary::to_python_wrapper() { return _godot_dictionary_to_python_wrapper(*this); }
PyObject *NodePath::to_python_wrapper() { return _nodepath_to_python_wrapper(*this); }
PyObject *Plane::to_python_wrapper() { return _plane_to_python_wrapper(*this); }
PyObject *PoolByteArray::to_python_wrapper() { return _poolbytearray_to_python_wrapper(*this); }
PyObject *PoolIntArray::to_python_wrapper() { return _poolintarray_to_python_wrapper(*this); }
PyObject *PoolRealArray::to_python_wrapper() { return _poolrealarray_to_python_wrapper(*this); }
PyObject *PoolStringArray::to_python_wrapper() { return _poolstringarray_to_python_wrapper(*this); }
PyObject *PoolVector2Array::to_python_wrapper() { return _poolvector2array_to_python_wrapper(*this); }
PyObject *PoolVector3Array::to_python_wrapper() { return _poolvector3array_to_python_wrapper(*this); }
PyObject *PoolColorArray::to_python_wrapper() { return _poolcolorarray_to_python_wrapper(*this); }
PyObject *Quat::to_python_wrapper() { return _quat_to_python_wrapper(*this); }
PyObject *Rect2::to_python_wrapper() { return _rect2_to_python_wrapper(*this); }
PyObject *RID::to_python_wrapper() { return _rid_to_python_wrapper(*this); }
PyObject *String::to_python_wrapper() { return _godot_string_to_python_wrapper(*this); }
PyObject *Transform::to_python_wrapper() { return _transform_to_python_wrapper(*this); }
PyObject *Transform2D::to_python_wrapper() { return _transform2d_to_python_wrapper(*this); }
PyObject *Vector2::to_python_wrapper() { return _vector2_to_python_wrapper(*this); }
PyObject *Vector3::to_python_wrapper() { return _vector3_to_python_wrapper(*this); }

Vector2 Vector2_from_PyObject(PyObject *obj) {
	if (Py_TYPE(obj) == PyGodotWrapperType_Vector2) {
		return *(Vector2 *)_python_wrapper_to_vector2(obj);
	}

	PYGODOT_CHECK_NUMPY_API();

	if (PyArray_Check(obj)) {
		return Vector2((PyArrayObject *)obj);
	}

	throw std::invalid_argument("incompatible Python object argument");
}

} // namespace godot
