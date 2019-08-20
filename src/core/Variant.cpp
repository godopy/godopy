#include "Variant.hpp"

#include <gdnative/variant.h>

#include "CoreTypes.hpp"
#include "Defs.hpp"
#include "GodotGlobal.hpp"
#include "Object.hpp"

#include <iostream>
#include <stdexcept>

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
	PYGODOT_CHECK_NUMPY_API();

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

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Vector2) {
		godot_vector2 *p = _python_wrapper_to_vector2((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_vector2(&_godot_variant, p);
		} else {
			godot::api->godot_variant_new_nil(&_godot_variant);
		}

	} else if (Py_TYPE(p_python_object) == PyGodotWrapperType_Array) {
		godot_array *p = _python_wrapper_to_godot_array((PyObject *)p_python_object);
		if (p) {
			godot::api->godot_variant_new_array(&_godot_variant, p);
		} else {
			godot::api->godot_variant_new_nil(&_godot_variant);
		}

		// TODO: Other Python wrappers
	} else if (PyArray_Check(p_python_object)) {
		PyArrayObject *arr = (PyArrayObject *)p_python_object;

		if (PyArray_NDIM(arr) == 1 && (PyArray_TYPE(arr) == NPY_FLOAT || PyArray_TYPE(arr) == NPY_DOUBLE)) {
			// 1-dimentional numeric arrays

			// TODO: Cast this to PoolRealArray

			if (PyArray_SIZE(arr) == 2) {
				Vector2 vec = Vector2(*(real_t *)PyArray_GETPTR1(arr, 0), *(real_t *)PyArray_GETPTR1(arr, 1));
				godot::api->godot_variant_new_vector2(&_godot_variant, (godot_vector2 *)&vec);
			} else if (PyArray_SIZE(arr) == 3) {
				Vector3 vec = Vector3(*(real_t *)PyArray_GETPTR1(arr, 0), *(real_t *)PyArray_GETPTR1(arr, 1), *(real_t *)PyArray_GETPTR1(arr, 2));
				godot::api->godot_variant_new_vector3(&_godot_variant, (godot_vector3 *)&vec);
			} else if (PyArray_SIZE(arr) == 4) {
				Quat q = Quat(*(real_t *)PyArray_GETPTR1(arr, 0), *(real_t *)PyArray_GETPTR1(arr, 1), *(real_t *)PyArray_GETPTR1(arr, 2), *(real_t *)PyArray_GETPTR1(arr, 3));
				godot::api->godot_variant_new_quat(&_godot_variant, (godot_quat *)&q);
			} else {
				// raises ValueError in Cython/Python context
				throw std::invalid_argument("required NumPy/Godot cast is not implemented yet");
			}
		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("required NumPy/Godot cast is not implemented yet");
		}
		// TODO: Other numpy arrays
	} else if (PyObject_IsInstance((PyObject *)p_python_object, (PyObject *)PyGodotType__Wrapped)) {
		godot_object *p = _cython_binding_to_godot_object((PyObject *)p_python_object);
		godot::api->godot_variant_new_object(&_godot_variant, p);

		// TODO: dict -> Dictionary, other iterables -> Array, array.Array -> PoolArray*, numpy.array -> PoolArray*

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
	godot_string s = godot::api->godot_variant_as_string(&_godot_variant);
	return *(String *)&s;
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
	godot_node_path s = godot::api->godot_variant_as_node_path(&_godot_variant);
	return *(NodePath *)&s;
}
Variant::operator RID() const {
	godot_rid s = godot::api->godot_variant_as_rid(&_godot_variant);
	return *(RID *)&s;
}

Variant::operator Dictionary() const {
	godot_dictionary d = godot::api->godot_variant_as_dictionary(&_godot_variant);
	return *(Dictionary *)&d;
}

Variant::operator Array() const {
	godot_array s = godot::api->godot_variant_as_array(&_godot_variant);
	return *(Array *)&s;
}

Variant::operator PoolByteArray() const {
	godot_pool_byte_array s = godot::api->godot_variant_as_pool_byte_array(&_godot_variant);
	return *(PoolByteArray *)&s;
}
Variant::operator PoolIntArray() const {
	godot_pool_int_array s = godot::api->godot_variant_as_pool_int_array(&_godot_variant);
	return *(PoolIntArray *)&s;
}
Variant::operator PoolRealArray() const {
	godot_pool_real_array s = godot::api->godot_variant_as_pool_real_array(&_godot_variant);
	return *(PoolRealArray *)&s;
}
Variant::operator PoolStringArray() const {
	godot_pool_string_array s = godot::api->godot_variant_as_pool_string_array(&_godot_variant);
	return *(PoolStringArray *)&s;
}
Variant::operator PoolVector2Array() const {
	godot_pool_vector2_array s = godot::api->godot_variant_as_pool_vector2_array(&_godot_variant);
	return *(PoolVector2Array *)&s;
}
Variant::operator PoolVector3Array() const {
	godot_pool_vector3_array s = godot::api->godot_variant_as_pool_vector3_array(&_godot_variant);
	return *(PoolVector3Array *)&s;
}
Variant::operator PoolColorArray() const {
	godot_pool_color_array s = godot::api->godot_variant_as_pool_color_array(&_godot_variant);
	return *(PoolColorArray *)&s;
}
Variant::operator godot_object *() const {
	return godot::api->godot_variant_as_object(&_godot_variant);
}

Variant::operator PyObject *() const {
	PyObject *obj;

	PYGODOT_CHECK_NUMPY_API();

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
			Vector2 vec = *this;
			const npy_intp dims[] = {2};
			obj = PyArray_SimpleNew(1, dims, NPY_FLOAT);

			real_t *p_x = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 0);
			real_t *p_y = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 1);
			*p_x = vec.x;
			*p_y = vec.y;
			break;
		}

		case VECTOR3: {
			Vector3 vec = *this;
			const npy_intp dims[] = {3};
			obj = PyArray_SimpleNew(1, dims, NPY_FLOAT);

			real_t *p_x = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 0);
			real_t *p_y = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 1);
			real_t *p_z = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 2);
			*p_x = vec.x;
			*p_y = vec.y;
			*p_z = vec.z;
			break;
		}

		case COLOR: {
			Color c = *this;
			const npy_intp dims[] = {4};
			obj = PyArray_SimpleNew(1, dims, NPY_FLOAT);

			real_t *p_r = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 0);
			real_t *p_g = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 1);
			real_t *p_b = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 2);
			real_t *p_a = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 3);
			*p_r = c.r;
			*p_g = c.g;
			*p_b = c.b;
			*p_a = c.a;
			break;
		}

		case QUAT: {
			Quat q = *this;
			const npy_intp dims[] = {4};
			obj = PyArray_SimpleNew(1, dims, NPY_FLOAT);

			real_t *p_x = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 0);
			real_t *p_y = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 1);
			real_t *p_z = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 2);
			real_t *p_w = (real_t *)PyArray_GETPTR1((PyArrayObject *)obj, 3);
			*p_x = q.x;
			*p_y = q.y;
			*p_z = q.z;
			*p_w = q.w;
			break;
		}

		// TODO: Add more convertions

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
