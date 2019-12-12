#include "Variant.hpp"

#include <gdnative/variant.h>

#include "CoreTypes.hpp"
#include "Defs.hpp"
#include "GodotGlobal.hpp"
#include "Object.hpp"

#include <iostream>
#include <stdexcept>

#ifndef NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#include "PythonGlobal.hpp"
#include <internal-packages/godot/core/types.hpp>

namespace godot {

Variant::Variant() {
	godot::api->godot_variant_new_nil(&_godot_variant);
}

Variant::Variant(const Variant &v) {
	godot::api->godot_variant_new_copy(&_godot_variant, &v._godot_variant);
}

Variant::Variant(bool p_bool) {
	godot::api->godot_variant_new_bool(&_godot_variant, p_bool);
}

Variant::Variant(signed int p_int) // real one
{
	godot::api->godot_variant_new_int(&_godot_variant, p_int);
}

Variant::Variant(unsigned int p_int) {
	godot::api->godot_variant_new_uint(&_godot_variant, p_int);
}

Variant::Variant(signed short p_short) // real one
{
	godot::api->godot_variant_new_int(&_godot_variant, (int)p_short);
}

Variant::Variant(int64_t p_char) // real one
{
	godot::api->godot_variant_new_int(&_godot_variant, p_char);
}

Variant::Variant(uint64_t p_char) {
	godot::api->godot_variant_new_uint(&_godot_variant, p_char);
}

Variant::Variant(float p_float) {
	godot::api->godot_variant_new_real(&_godot_variant, p_float);
}

Variant::Variant(double p_double) {
	godot::api->godot_variant_new_real(&_godot_variant, p_double);
}

Variant::Variant(const String &p_string) {
	godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&p_string);
}

Variant::Variant(const char *const p_cstring) {
	String s = String(p_cstring);
	godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&s);
}

Variant::Variant(const wchar_t *p_wstring) {
	String s = p_wstring;
	godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&s);
}

Variant::Variant(const Vector2 &p_vector2) {
	godot::api->godot_variant_new_vector2(&_godot_variant, (godot_vector2 *)&p_vector2);
}

Variant::Variant(const Rect2 &p_rect2) {
	godot::api->godot_variant_new_rect2(&_godot_variant, (godot_rect2 *)&p_rect2);
}

Variant::Variant(const Vector3 &p_vector3) {
	godot::api->godot_variant_new_vector3(&_godot_variant, (godot_vector3 *)&p_vector3);
}

Variant::Variant(const Plane &p_plane) {
	godot::api->godot_variant_new_plane(&_godot_variant, (godot_plane *)&p_plane);
}

Variant::Variant(const AABB &p_aabb) {
	godot::api->godot_variant_new_aabb(&_godot_variant, (godot_aabb *)&p_aabb);
}

Variant::Variant(const Quat &p_quat) {
	godot::api->godot_variant_new_quat(&_godot_variant, (godot_quat *)&p_quat);
}

Variant::Variant(const Basis &p_transform) {
	godot::api->godot_variant_new_basis(&_godot_variant, (godot_basis *)&p_transform);
}

Variant::Variant(const Transform2D &p_transform) {
	godot::api->godot_variant_new_transform2d(&_godot_variant, (godot_transform2d *)&p_transform);
}

Variant::Variant(const Transform &p_transform) {
	godot::api->godot_variant_new_transform(&_godot_variant, (godot_transform *)&p_transform);
}

Variant::Variant(const Color &p_color) {
	godot::api->godot_variant_new_color(&_godot_variant, (godot_color *)&p_color);
}

Variant::Variant(const NodePath &p_path) {
	godot::api->godot_variant_new_node_path(&_godot_variant, (godot_node_path *)&p_path);
}

Variant::Variant(const RID &p_rid) {
	godot::api->godot_variant_new_rid(&_godot_variant, (godot_rid *)&p_rid);
}

Variant::Variant(const Object *p_object) {
	if (p_object)
		godot::api->godot_variant_new_object(&_godot_variant, p_object->_owner);
	else
		godot::api->godot_variant_new_nil(&_godot_variant);
}

Variant::Variant(const Dictionary &p_dictionary) {
	godot::api->godot_variant_new_dictionary(&_godot_variant, (godot_dictionary *)&p_dictionary);
}

Variant::Variant(const Array &p_array) {
	godot::api->godot_variant_new_array(&_godot_variant, (godot_array *)&p_array);
}

Variant::Variant(const PoolByteArray &p_raw_array) {
	godot::api->godot_variant_new_pool_byte_array(&_godot_variant, (godot_pool_byte_array *)&p_raw_array);
}

Variant::Variant(const PoolIntArray &p_int_array) {
	godot::api->godot_variant_new_pool_int_array(&_godot_variant, (godot_pool_int_array *)&p_int_array);
}

Variant::Variant(const PoolRealArray &p_real_array) {
	godot::api->godot_variant_new_pool_real_array(&_godot_variant, (godot_pool_real_array *)&p_real_array);
}

Variant::Variant(const PoolStringArray &p_string_array) {
	godot::api->godot_variant_new_pool_string_array(&_godot_variant, (godot_pool_string_array *)&p_string_array);
}

Variant::Variant(const PoolVector2Array &p_vector2_array) {
	godot::api->godot_variant_new_pool_vector2_array(&_godot_variant, (godot_pool_vector2_array *)&p_vector2_array);
}

Variant::Variant(const PoolVector3Array &p_vector3_array) {
	godot::api->godot_variant_new_pool_vector3_array(&_godot_variant, (godot_pool_vector3_array *)&p_vector3_array);
}

Variant::Variant(const PoolColorArray &p_color_array) {
	godot::api->godot_variant_new_pool_color_array(&_godot_variant, (godot_pool_color_array *)&p_color_array);
}

Variant::Variant(const PyObject *p_python_object) {
	if (p_python_object == Py_None) {
		// Py_XDECREF(p_python_object); // XXX
		godot::api->godot_variant_new_nil(&_godot_variant);

	} else if (PyBool_Check(p_python_object)) {
		godot::api->godot_variant_new_bool(&_godot_variant, PyLong_AsLong((PyObject *)p_python_object));

	} else if (PyLong_Check(p_python_object)) {
		godot::api->godot_variant_new_int(&_godot_variant, PyLong_AsLong((PyObject *)p_python_object));

	} else if (PyFloat_Check(p_python_object)) {
		const double p_double = PyFloat_AsDouble((PyObject *)p_python_object);
		godot::api->godot_variant_new_real(&_godot_variant, p_double);

	} else if (PyUnicode_Check(p_python_object) || PyBytes_Check(p_python_object)) {
		String s = String(p_python_object);
		godot::api->godot_variant_new_string(&_godot_variant, (godot_string *)&s);

	} else if (PyByteArray_Check(p_python_object)) {
		godot_pool_byte_array *p;
		godot::api->godot_pool_byte_array_new(p);
		godot::api->godot_pool_byte_array_resize(p, PyByteArray_GET_SIZE(p_python_object));
		godot_pool_byte_array_write_access *_write_access = godot::api->godot_pool_byte_array_write(p);

		const uint8_t *ptr = godot::api->godot_pool_byte_array_write_access_ptr(_write_access);
		memcpy((void *)ptr, (void *)PyByteArray_AS_STRING(p_python_object), PyByteArray_GET_SIZE(p_python_object));

		godot::api->godot_variant_new_pool_byte_array(&_godot_variant, p);

		godot::api->godot_pool_byte_array_write_access_destroy(_write_access);
		godot::api->godot_pool_byte_array_destroy(p);

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_AABB) {
		godot_aabb *p = _python_wrapper_to_aabb((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_aabb(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Array) {
		godot_array *p = _python_wrapper_to_godot_array((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_array(&_godot_variant, p);
		} else {
			godot::api->godot_variant_new_nil(&_godot_variant);
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Basis) {
		godot_basis *p = _python_wrapper_to_basis((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_basis(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Color) {
		godot_color *p = _python_wrapper_to_color((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_color(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Dictionary) {
		godot_dictionary *p = _python_wrapper_to_godot_dictionary((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_dictionary(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_NodePath) {
		godot_node_path *p = _python_wrapper_to_nodepath((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_node_path(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Plane) {
		godot_plane *p = _python_wrapper_to_plane((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_plane(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolByteArray) {
		godot_pool_byte_array *p = _python_wrapper_to_poolbytearray((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_byte_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolIntArray) {
		godot_pool_int_array *p = _python_wrapper_to_poolintarray((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_int_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolRealArray) {
		godot_pool_real_array *p = _python_wrapper_to_poolrealarray((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_real_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolStringArray) {
		godot_pool_string_array *p = _python_wrapper_to_poolstringarray((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_string_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolVector2Array) {
		godot_pool_vector2_array *p = _python_wrapper_to_poolvector2array((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_vector2_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolVector3Array) {
		godot_pool_vector3_array *p = _python_wrapper_to_poolvector3array((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_vector3_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_PoolColorArray) {
		godot_pool_color_array *p = _python_wrapper_to_poolcolorarray((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_pool_color_array(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Quat) {
		godot_quat *p = _python_wrapper_to_quat((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_quat(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Rect2) {
		godot_rect2 *p = _python_wrapper_to_rect2((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_rect2(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_RID) {
		godot_rid *p = _python_wrapper_to_rid((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_rid(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_String) {
		godot_string *p = _python_wrapper_to_godot_string((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_string(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Transform) {
		godot_transform *p = _python_wrapper_to_transform((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_transform(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Transform2D) {
		godot_transform2d *p = _python_wrapper_to_transform2d((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_transform2d(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Vector2) {
		godot_vector2 *p = _python_wrapper_to_vector2((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_vector2(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Vector3) {
		godot_vector3 *p = _python_wrapper_to_vector3((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_vector3(&_godot_variant, p);
		} else {
			throw std::invalid_argument("could not convert Python object to Variant");
		}

	} else if (PyObject_IsInstance((PyObject *)p_python_object, (PyObject *)PyGodotType__Wrapped)) {
		godot_object *p = _cython_binding_to_godot_object((PyObject *)p_python_object);
		godot::api->godot_variant_new_object(&_godot_variant, p);

		// TODO: dict -> Dictionary, other iterables -> Array, array.Array -> PoolArray*, numpy.array -> PoolArray*

	} else if (PyArray_Check((PyObject *)p_python_object)) {
		PyArrayObject *arr = (PyArrayObject *)p_python_object;

		if (PyArray_NDIM(arr) == 1 && PyArray_TYPE(arr) == NPY_UINT8) {
			PoolByteArray _arr = PoolByteArray(arr);
			godot::api->godot_variant_new_pool_byte_array(&_godot_variant, (godot_pool_byte_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 1 && PyArray_ISINTEGER(arr)) {
			PoolIntArray _arr(arr);
			godot::api->godot_variant_new_pool_int_array(&_godot_variant, (godot_pool_int_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 1 && PyArray_ISFLOAT(arr)) {
			PoolRealArray _arr(arr);
			godot::api->godot_variant_new_pool_real_array(&_godot_variant, (godot_pool_real_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 1 && PyArray_ISSTRING(arr)) {
			PoolStringArray _arr(arr);
			godot::api->godot_variant_new_pool_string_array(&_godot_variant, (godot_pool_string_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 1) {
			Array _arr;

			for (int idx = 0; idx < PyArray_SIZE(arr); idx++) {
				PyObject *item = PyArray_GETITEM(arr, (const char *)PyArray_GETPTR1(arr, idx));
				// TODO: Check NULL pointer
				_arr.append(Variant(item));
			}

			godot::api->godot_variant_new_array(&_godot_variant, (godot_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 2 && PyArray_ISNUMBER(arr) && PyArray_DIM(arr, 1) == 2) {
			PoolVector2Array _arr = PoolVector2Array(arr);
			godot::api->godot_variant_new_pool_vector2_array(&_godot_variant, (godot_pool_vector2_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 2 && PyArray_ISNUMBER(arr) && PyArray_DIM(arr, 1) == 3) {
			PoolVector3Array _arr = PoolVector3Array(arr);
			godot::api->godot_variant_new_pool_vector3_array(&_godot_variant, (godot_pool_vector3_array *)&_arr);

		} else if (PyArray_NDIM(arr) == 2 && PyArray_ISNUMBER(arr) && PyArray_DIM(arr, 1) == 4) {
			PoolColorArray _arr = PoolColorArray(arr);
			godot::api->godot_variant_new_pool_color_array(&_godot_variant, (godot_pool_color_array *)&_arr);

		} else {
			throw std::invalid_argument("could not convert NumPy array");
		}

	} else if (PySequence_Check((PyObject *)p_python_object)) {
		Array arr = Array(p_python_object);
		godot::api->godot_variant_new_array(&_godot_variant, (godot_array *)&arr);

	} else if (PyMapping_Check((PyObject *)p_python_object)) {
		Dictionary dict = Dictionary(p_python_object);
		godot::api->godot_variant_new_dictionary(&_godot_variant, (godot_dictionary *)&dict);

	} else if (PyIndex_Check((PyObject *)p_python_object)) {
		const Py_ssize_t p_num = PyNumber_AsSsize_t((PyObject *)p_python_object, NULL);
		godot::api->godot_variant_new_int(&_godot_variant, p_num);

	} else if (PyNumber_Check((PyObject *)p_python_object)) {
		PyObject *p_num = PyNumber_Float((PyObject *)p_python_object);
		const double p_double = PyFloat_AsDouble((PyObject *)p_num);
		godot::api->godot_variant_new_real(&_godot_variant, p_double);

	} else if (PyObject_CheckBuffer((PyObject *)p_python_object)) {
		throw std::invalid_argument("generic Python buffer support is not implemented yet");

	} else {

		// raises ValueError in Cython/Python context
		throw std::invalid_argument("could not cast Python object to Godot Variant");
	}
}

Variant &Variant::operator=(const Variant &v) {
	godot::api->godot_variant_new_copy(&_godot_variant, &v._godot_variant);
	return *this;
}

Variant::operator bool() const {
	return booleanize();
}
Variant::operator signed int() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator unsigned int() const // this is the real one
{
	return godot::api->godot_variant_as_uint(&_godot_variant);
}
Variant::operator signed short() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator unsigned short() const {
	return godot::api->godot_variant_as_uint(&_godot_variant);
}
Variant::operator signed char() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator unsigned char() const {
	return godot::api->godot_variant_as_uint(&_godot_variant);
}
Variant::operator int64_t() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}
Variant::operator uint64_t() const {
	return godot::api->godot_variant_as_uint(&_godot_variant);
}

Variant::operator wchar_t() const {
	return godot::api->godot_variant_as_int(&_godot_variant);
}

Variant::operator float() const {
	return godot::api->godot_variant_as_real(&_godot_variant);
}

Variant::operator double() const {
	return godot::api->godot_variant_as_real(&_godot_variant);
}
Variant::operator String() const {
	String ret;
	*(godot_string *)&ret = godot::api->godot_variant_as_string(&_godot_variant);
	return ret;
}
Variant::operator Vector2() const {
	godot_vector2 s = godot::api->godot_variant_as_vector2(&_godot_variant);
	return *(Vector2 *)&s;
}
Variant::operator Rect2() const {
	godot_rect2 s = godot::api->godot_variant_as_rect2(&_godot_variant);
	return *(Rect2 *)&s;
}
Variant::operator Vector3() const {
	godot_vector3 s = godot::api->godot_variant_as_vector3(&_godot_variant);
	return *(Vector3 *)&s;
}
Variant::operator Plane() const {
	godot_plane s = godot::api->godot_variant_as_plane(&_godot_variant);
	return *(Plane *)&s;
}
Variant::operator AABB() const {
	godot_aabb s = godot::api->godot_variant_as_aabb(&_godot_variant);
	return *(AABB *)&s;
}
Variant::operator Quat() const {
	godot_quat s = godot::api->godot_variant_as_quat(&_godot_variant);
	return *(Quat *)&s;
}
Variant::operator Basis() const {
	godot_basis s = godot::api->godot_variant_as_basis(&_godot_variant);
	return *(Basis *)&s;
}
Variant::operator Transform() const {
	godot_transform s = godot::api->godot_variant_as_transform(&_godot_variant);
	return *(Transform *)&s;
}
Variant::operator Transform2D() const {
	godot_transform2d s = godot::api->godot_variant_as_transform2d(&_godot_variant);
	return *(Transform2D *)&s;
}

Variant::operator Color() const {
	godot_color s = godot::api->godot_variant_as_color(&_godot_variant);
	return *(Color *)&s;
}
Variant::operator NodePath() const {
	NodePath ret;
	*(godot_node_path *)&ret = godot::api->godot_variant_as_node_path(&_godot_variant);
	return ret;
}
Variant::operator RID() const {
	godot_rid s = godot::api->godot_variant_as_rid(&_godot_variant);
	return *(RID *)&s;
}

Variant::operator Dictionary() const {
	Dictionary ret;
	*(godot_dictionary *)&ret = godot::api->godot_variant_as_dictionary(&_godot_variant);
	return ret;
}

Variant::operator Array() const {
	Array ret;
	*(godot_array *)&ret = godot::api->godot_variant_as_array(&_godot_variant);
	return ret;
}

Variant::operator PoolByteArray() const {
	PoolByteArray ret;
	*(godot_pool_byte_array *)&ret = godot::api->godot_variant_as_pool_byte_array(&_godot_variant);
	return ret;
}
Variant::operator PoolIntArray() const {
	PoolIntArray ret;
	*(godot_pool_int_array *)&ret = godot::api->godot_variant_as_pool_int_array(&_godot_variant);
	return ret;
}
Variant::operator PoolRealArray() const {
	PoolRealArray ret;
	*(godot_pool_real_array *)&ret = godot::api->godot_variant_as_pool_real_array(&_godot_variant);
	return ret;
}
Variant::operator PoolStringArray() const {
	PoolStringArray ret;
	*(godot_pool_string_array *)&ret = godot::api->godot_variant_as_pool_string_array(&_godot_variant);
	return ret;
}
Variant::operator PoolVector2Array() const {
	PoolVector2Array ret;
	*(godot_pool_vector2_array *)&ret = godot::api->godot_variant_as_pool_vector2_array(&_godot_variant);
	return ret;
}
Variant::operator PoolVector3Array() const {
	PoolVector3Array ret;
	*(godot_pool_vector3_array *)&ret = godot::api->godot_variant_as_pool_vector3_array(&_godot_variant);
	return ret;
}
Variant::operator PoolColorArray() const {
	PoolColorArray ret;
	*(godot_pool_color_array *)&ret = godot::api->godot_variant_as_pool_color_array(&_godot_variant);
	return ret;
}
Variant::operator godot_object *() const {
	return godot::api->godot_variant_as_object(&_godot_variant);
}

Variant::operator PyObject *() const {
	PyObject *obj;

	switch (get_type()) {
		case NIL:
			obj = Py_None;
			break;

		case BOOL:
			obj = booleanize() ? Py_True : Py_False;
			break;

		case INT:
			obj = PyLong_FromSsize_t(godot::api->godot_variant_as_int(&_godot_variant));
			break;

		case REAL:
			obj = PyFloat_FromDouble(godot::api->godot_variant_as_real(&_godot_variant));
			break;

		case STRING: {
			String s = *this;
			obj = PyUnicode_FromWideChar(s.unicode_str(), s.length());
			break;
		}

		case VECTOR2: {
			Vector2 cpp_obj = *this;
			return _vector2_to_python_wrapper(cpp_obj);
		}

		case RECT2: {
			Rect2 cpp_obj = *this;
			return _rect2_to_python_wrapper(cpp_obj);
		}

		case VECTOR3: {
			Vector3 cpp_obj = *this;
			return _vector3_to_python_wrapper(cpp_obj);
		}

		case TRANSFORM2D: {
			Transform2D cpp_obj = *this;
			return _transform2d_to_python_wrapper(cpp_obj);
		}

		case PLANE: {
			Plane cpp_obj = *this;
			return _plane_to_python_wrapper(cpp_obj);
		}

		case QUAT: {
			Quat cpp_obj = *this;
			return _quat_to_python_wrapper(cpp_obj);
		}

		case RECT3: {
			AABB cpp_obj = *this;
			return _aabb_to_python_wrapper(cpp_obj);
		}

		case BASIS: {
			Basis cpp_obj = *this;
			return _basis_to_python_wrapper(cpp_obj);
		}

		case TRANSFORM: {
			Transform cpp_obj = *this;
			return _transform_to_python_wrapper(cpp_obj);
		}

		case COLOR: {
			Color cpp_obj = *this;
			return _color_to_python_wrapper(cpp_obj);
		}

		case NODE_PATH: {
			NodePath cpp_obj = *this;
			return _nodepath_to_python_wrapper(cpp_obj);
		}

		case _RID: {
			RID cpp_obj = *this;
			return _rid_to_python_wrapper(cpp_obj);
		}

		case OBJECT: {
			godot_object *c_obj = godot::api->godot_variant_as_object(&_godot_variant);
			return _godot_object_to_python_binding(c_obj);
		}

		case DICTIONARY: {
			const Dictionary dict = *this;
			const Array keys = dict.keys();
			obj = PyDict_New();

			for (int i = 0; i < keys.size(); i++) {
				Variant _key = keys[i];
				PyObject *key = _key;
				PyObject *val = dict[_key];
				// TODO: Check unlikely NULL pointers
				PyDict_SetItem(obj, key, val);
			}
			break;
		}

		case ARRAY: {
			const Array arr = *this;
			obj = PyTuple_New(arr.size());

			for (int i = 0; i < arr.size(); i++) {
				PyObject *item = arr[i];
				// TODO: Check unlikely NULL pointers
				PyTuple_SET_ITEM(obj, i, item);
			}
			break;
		}

		case POOL_BYTE_ARRAY: {
			PoolByteArray cpp_obj = *this;
			return _poolbytearray_to_python_wrapper(cpp_obj);
		}

		case POOL_INT_ARRAY: {
			PoolIntArray cpp_obj = *this;
			return _poolintarray_to_python_wrapper(cpp_obj);
		}

		case POOL_REAL_ARRAY: {
			PoolRealArray cpp_obj = *this;
			return _poolrealarray_to_python_wrapper(cpp_obj);
		}

		case POOL_STRING_ARRAY: {
			PoolStringArray cpp_obj = *this;
			return _poolstringarray_to_numpy(cpp_obj);
		}

		case POOL_VECTOR2_ARRAY: {
			PoolVector2Array cpp_obj = *this;
			return _poolvector2array_to_python_wrapper(cpp_obj);
		}

		case POOL_VECTOR3_ARRAY: {
			PoolVector3Array cpp_obj = *this;
			return _poolvector3array_to_python_wrapper(cpp_obj);
		}

		case POOL_COLOR_ARRAY: {
			PoolColorArray cpp_obj = *this;
			return _poolcolorarray_to_python_wrapper(cpp_obj);
		}

		default:
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("could not cast Python object to Godot Variant");
	}

	Py_XINCREF(obj);
	return obj;
}

Variant::Type Variant::get_type() const {
	return (Type)godot::api->godot_variant_get_type(&_godot_variant);
}

Variant Variant::call(const String &method, const Variant **args, const int arg_count) {
	Variant v;
	*(godot_variant *)&v = godot::api->godot_variant_call(&_godot_variant, (godot_string *)&method, (const godot_variant **)args, arg_count, nullptr);
	return v;
}

bool Variant::has_method(const String &method) {
	return godot::api->godot_variant_has_method(&_godot_variant, (godot_string *)&method);
}

bool Variant::operator==(const Variant &b) const {
	return godot::api->godot_variant_operator_equal(&_godot_variant, &b._godot_variant);
}

bool Variant::operator!=(const Variant &b) const {
	return !(*this == b);
}

bool Variant::operator<(const Variant &b) const {
	return godot::api->godot_variant_operator_less(&_godot_variant, &b._godot_variant);
}

bool Variant::operator<=(const Variant &b) const {
	return (*this < b) || (*this == b);
}

bool Variant::operator>(const Variant &b) const {
	return !(*this <= b);
}

bool Variant::operator>=(const Variant &b) const {
	return !(*this < b);
}

bool Variant::hash_compare(const Variant &b) const {
	return godot::api->godot_variant_hash_compare(&_godot_variant, &b._godot_variant);
}

bool Variant::booleanize() const {
	return godot::api->godot_variant_booleanize(&_godot_variant);
}

Variant::~Variant() {
	godot::api->godot_variant_destroy(&_godot_variant);
}

} // namespace godot
