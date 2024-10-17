/**************************************************************************/
/*  variant.cpp                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/godopy.hpp>

#include <binding.h>

#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/defs.hpp>

#include <utility>

namespace godot {

GDExtensionVariantFromTypeConstructorFunc Variant::from_type_constructor[Variant::VARIANT_MAX]{};
GDExtensionTypeFromVariantConstructorFunc Variant::to_type_constructor[Variant::VARIANT_MAX]{};

void Variant::init_bindings() {
	// Start from 1 to skip NIL.
	for (int i = 1; i < VARIANT_MAX; i++) {
		from_type_constructor[i] = internal::gdextension_interface_get_variant_from_type_constructor((GDExtensionVariantType)i);
		to_type_constructor[i] = internal::gdextension_interface_get_variant_to_type_constructor((GDExtensionVariantType)i);
	}

	StringName::init_bindings();
	String::init_bindings();
	NodePath::init_bindings();
	RID::init_bindings();
	Callable::init_bindings();
	Signal::init_bindings();
	Dictionary::init_bindings();
	Array::init_bindings();
	PackedByteArray::init_bindings();
	PackedInt32Array::init_bindings();
	PackedInt64Array::init_bindings();
	PackedFloat32Array::init_bindings();
	PackedFloat64Array::init_bindings();
	PackedStringArray::init_bindings();
	PackedVector2Array::init_bindings();
	PackedVector3Array::init_bindings();
	PackedVector4Array::init_bindings();
	PackedColorArray::init_bindings();
}

Variant::Variant() {
	internal::gdextension_interface_variant_new_nil(_native_ptr());
}

Variant::Variant(GDExtensionConstVariantPtr native_ptr) {
	internal::gdextension_interface_variant_new_copy(_native_ptr(), native_ptr);
}

Variant::Variant(const Variant &other) {
	internal::gdextension_interface_variant_new_copy(_native_ptr(), other._native_ptr());
}

Variant::Variant(Variant &&other) {
	std::swap(opaque, other.opaque);
}

Variant::Variant(bool v) {
	GDExtensionBool encoded;
	PtrToArg<bool>::encode(v, &encoded);
	from_type_constructor[BOOL](_native_ptr(), &encoded);
}

Variant::Variant(int64_t v) {
	GDExtensionInt encoded;
	PtrToArg<int64_t>::encode(v, &encoded);
	from_type_constructor[INT](_native_ptr(), &encoded);
}

Variant::Variant(double v) {
	double encoded;
	PtrToArg<double>::encode(v, &encoded);
	from_type_constructor[FLOAT](_native_ptr(), &encoded);
}

Variant::Variant(const String &v) {
	from_type_constructor[STRING](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Vector2 &v) {
	from_type_constructor[VECTOR2](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector2i &v) {
	from_type_constructor[VECTOR2I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Rect2 &v) {
	from_type_constructor[RECT2](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Rect2i &v) {
	from_type_constructor[RECT2I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector3 &v) {
	from_type_constructor[VECTOR3](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector3i &v) {
	from_type_constructor[VECTOR3I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Transform2D &v) {
	from_type_constructor[TRANSFORM2D](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector4 &v) {
	from_type_constructor[VECTOR4](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector4i &v) {
	from_type_constructor[VECTOR4I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Plane &v) {
	from_type_constructor[PLANE](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Quaternion &v) {
	from_type_constructor[QUATERNION](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const godot::AABB &v) {
	from_type_constructor[AABB](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Basis &v) {
	from_type_constructor[BASIS](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Transform3D &v) {
	from_type_constructor[TRANSFORM3D](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Projection &v) {
	from_type_constructor[PROJECTION](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Color &v) {
	from_type_constructor[COLOR](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const StringName &v) {
	from_type_constructor[STRING_NAME](_native_ptr(), v._native_ptr());
}

Variant::Variant(const NodePath &v) {
	from_type_constructor[NODE_PATH](_native_ptr(), v._native_ptr());
}

Variant::Variant(const godot::RID &v) {
	from_type_constructor[RID](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Object *v) {
	if (v) {
		from_type_constructor[OBJECT](_native_ptr(), const_cast<GodotObject **>(&v->_owner));
	} else {
		GodotObject *nullobject = nullptr;
		from_type_constructor[OBJECT](_native_ptr(), &nullobject);
	}
}

Variant::Variant(const ObjectID &p_id) :
		Variant(p_id.operator uint64_t()) {
}

Variant::Variant(const Callable &v) {
	from_type_constructor[CALLABLE](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Signal &v) {
	from_type_constructor[SIGNAL](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Dictionary &v) {
	from_type_constructor[DICTIONARY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Array &v) {
	from_type_constructor[ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedByteArray &v) {
	from_type_constructor[PACKED_BYTE_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedInt32Array &v) {
	from_type_constructor[PACKED_INT32_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedInt64Array &v) {
	from_type_constructor[PACKED_INT64_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedFloat32Array &v) {
	from_type_constructor[PACKED_FLOAT32_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedFloat64Array &v) {
	from_type_constructor[PACKED_FLOAT64_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedStringArray &v) {
	from_type_constructor[PACKED_STRING_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedVector2Array &v) {
	from_type_constructor[PACKED_VECTOR2_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedVector3Array &v) {
	from_type_constructor[PACKED_VECTOR3_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedColorArray &v) {
	from_type_constructor[PACKED_COLOR_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedVector4Array &v) {
	from_type_constructor[PACKED_VECTOR4_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PyObject *v_const) {
	// IMPORTANT: Should be called only with GIL! Responsibility is on the caller

	ERR_FAIL_NULL(v_const);

	PyObject *v = const_cast<PyObject *>(v_const);

	if (v == Py_None) {
		internal::gdextension_interface_variant_new_nil(_native_ptr());

	} else if (PyBool_Check(v)) {
		GDExtensionBool encoded;
		if (v == Py_True) {
			PtrToArg<bool>::encode(true, &encoded);
		} else {
			PtrToArg<bool>::encode(false, &encoded);
		}

		from_type_constructor[BOOL](_native_ptr(), &encoded);

	} else if (PyLong_Check(v)) {
		GDExtensionInt encoded;
		PtrToArg<int64_t>::encode(PyLong_AsSize_t(v), &encoded);

		from_type_constructor[INT](_native_ptr(), &encoded);

	} else if (PyFloat_Check(v)) {
		double encoded;
		PtrToArg<double>::encode(PyFloat_AsDouble(v), &encoded);

		from_type_constructor[FLOAT](_native_ptr(), &encoded);

	} else if (PyUnicode_Check(v) || PyBytes_Check(v)) {
		String s = String(v);
		from_type_constructor[STRING](_native_ptr(), s._native_ptr());

	} else if (PyByteArray_Check(v) || PyObject_CheckBuffer(v)) {
		// TODO: Check for various numpy arrays first and create types accordingly

		PackedByteArray a = PackedByteArray();
		Py_buffer *view = nullptr;
		uint8_t *buf;
		int result = PyObject_GetBuffer(v, view, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS);

		if (result == 0 && view != nullptr) {
			buf = (uint8_t *)view->buf;
			a.resize(view->len);
			for (size_t i = 0; i < view->len; i++) {
				a[i] = buf[i];
			}

			PyBuffer_Release(view);
		}

		from_type_constructor[PACKED_BYTE_ARRAY](_native_ptr(), a._native_ptr());

	} else if (PySequence_Check(v)) {
		Array a = Array();
		Py_ssize_t size = PySequence_Size(v);
		a.resize(size);

		PyObject *item;
		Variant _item;

		for (size_t i = 0; i < size; i++) {
			item = PySequence_GetItem(v, i);
			_item = Variant(item);
			a[i] = _item;
		}

		from_type_constructor[ARRAY](_native_ptr(), a._native_ptr());

	} else if (PyMapping_Check(v)) {
		Dictionary d = Dictionary();
		PyObject *keys = PyMapping_Keys(v);
		Py_ssize_t size = PySequence_Size(keys);

		PyObject *key;
		Variant _key;
		PyObject *value;
		Variant _value;

		for (size_t i = 0; i < size; i++) {
			key = PySequence_GetItem(keys, i);
			_key = Variant(key);
			value = PyObject_GetItem(v, key);
			_value = Variant(value);
			d[_key] = _value;
		}

		from_type_constructor[DICTIONARY](_native_ptr(), d._native_ptr());

	} else if (PyIndex_Check(v)) {
		GDExtensionInt encoded;
		PtrToArg<int64_t>::encode(PyNumber_AsSsize_t(v, NULL), &encoded);
		from_type_constructor[INT](_native_ptr(), &encoded);

	} else if (PyNumber_Check(v)) {
		PyObject *number = PyNumber_Float(v);
		ERR_FAIL_NULL(number);
		double encoded;
		PtrToArg<double>::encode(PyFloat_AsDouble(number), &encoded);
		from_type_constructor[FLOAT](_native_ptr(), &encoded);

	} else if (PyObject_IsInstance(v, (PyObject *)&GDPy_ObjectType)) {
		from_type_constructor[OBJECT](_native_ptr(), &((GDPy_Object *)v)->_owner);

	} else {
		internal::gdextension_interface_variant_new_nil(_native_ptr());
		ERR_PRINT("NOT IMPLEMENTED: Could not cast Python object to Godot Variant. "
				  "Unsupported or unknown Python object.");
	}
}

Variant::~Variant() {
	internal::gdextension_interface_variant_destroy(_native_ptr());
}

Variant::operator PyObject *() const {
	return pythonize();
}

Variant::operator bool() const {
	GDExtensionBool result;
	to_type_constructor[BOOL](&result, _native_ptr());
	return PtrToArg<bool>::convert(&result);
}

Variant::operator int64_t() const {
	GDExtensionInt result;
	to_type_constructor[INT](&result, _native_ptr());
	return PtrToArg<int64_t>::convert(&result);
}

Variant::operator int32_t() const {
	return static_cast<int32_t>(operator int64_t());
}

Variant::operator int16_t() const {
	return static_cast<int16_t>(operator int64_t());
}

Variant::operator int8_t() const {
	return static_cast<int8_t>(operator int64_t());
}

Variant::operator uint64_t() const {
	return static_cast<uint64_t>(operator int64_t());
}

Variant::operator uint32_t() const {
	return static_cast<uint32_t>(operator int64_t());
}

Variant::operator uint16_t() const {
	return static_cast<uint16_t>(operator int64_t());
}

Variant::operator uint8_t() const {
	return static_cast<uint8_t>(operator int64_t());
}

Variant::operator double() const {
	double result;
	to_type_constructor[FLOAT](&result, _native_ptr());
	return PtrToArg<double>::convert(&result);
}

Variant::operator float() const {
	return static_cast<float>(operator double());
}

Variant::operator String() const {
	return String(this);
}

Variant::operator Vector2() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector2 result;
	to_type_constructor[VECTOR2]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector2i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector2i result;
	to_type_constructor[VECTOR2I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Rect2() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Rect2 result;
	to_type_constructor[RECT2]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Rect2i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Rect2i result;
	to_type_constructor[RECT2I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector3() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector3 result;
	to_type_constructor[VECTOR3]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector3i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector3i result;
	to_type_constructor[VECTOR3I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Transform2D() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Transform2D result;
	to_type_constructor[TRANSFORM2D]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector4() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector4 result;
	to_type_constructor[VECTOR4]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector4i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector4i result;
	to_type_constructor[VECTOR4I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Plane() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Plane result;
	to_type_constructor[PLANE]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Quaternion() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Quaternion result;
	to_type_constructor[QUATERNION]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator godot::AABB() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	godot::AABB result;
	to_type_constructor[AABB]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Basis() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Basis result;
	to_type_constructor[BASIS]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Transform3D() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Transform3D result;
	to_type_constructor[TRANSFORM3D]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Projection() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Projection result;
	to_type_constructor[PROJECTION]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Color() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Color result;
	to_type_constructor[COLOR]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator StringName() const {
	return StringName(this);
}

Variant::operator NodePath() const {
	return NodePath(this);
}

Variant::operator godot::RID() const {
	return godot::RID(this);
}

Variant::operator Object *() const {
	GodotObject *obj;
	to_type_constructor[OBJECT](&obj, _native_ptr());
	if (obj == nullptr) {
		return nullptr;
	}
	return internal::get_object_instance_binding(obj);
}

Variant::operator ObjectID() const {
	if (get_type() == Type::INT) {
		return ObjectID(operator uint64_t());
	} else if (get_type() == Type::OBJECT) {
		Object *obj = operator Object *();
		if (obj != nullptr) {
			return ObjectID(obj->get_instance_id());
		} else {
			return ObjectID();
		}
	} else {
		return ObjectID();
	}
}

Variant::operator Callable() const {
	return Callable(this);
}

Variant::operator Signal() const {
	return Signal(this);
}

Variant::operator Dictionary() const {
	return Dictionary(this);
}

Variant::operator Array() const {
	return Array(this);
}

Variant::operator PackedByteArray() const {
	return PackedByteArray(this);
}

Variant::operator PackedInt32Array() const {
	return PackedInt32Array(this);
}

Variant::operator PackedInt64Array() const {
	return PackedInt64Array(this);
}

Variant::operator PackedFloat32Array() const {
	return PackedFloat32Array(this);
}

Variant::operator PackedFloat64Array() const {
	return PackedFloat64Array(this);
}

Variant::operator PackedStringArray() const {
	return PackedStringArray(this);
}

Variant::operator PackedVector2Array() const {
	return PackedVector2Array(this);
}

Variant::operator PackedVector3Array() const {
	return PackedVector3Array(this);
}

Variant::operator PackedColorArray() const {
	return PackedColorArray(this);
}

Variant::operator PackedVector4Array() const {
	return PackedVector4Array(this);
}

Variant &Variant::operator=(const Variant &other) {
	clear();
	internal::gdextension_interface_variant_new_copy(_native_ptr(), other._native_ptr());
	return *this;
}

Variant &Variant::operator=(Variant &&other) {
	std::swap(opaque, other.opaque);
	return *this;
}

bool Variant::operator==(const Variant &other) const {
	if (get_type() != other.get_type()) {
		return false;
	}
	bool valid = false;
	Variant result;
	evaluate(OP_EQUAL, *this, other, result, valid);
	return result.operator bool();
}

bool Variant::operator!=(const Variant &other) const {
	if (get_type() != other.get_type()) {
		return true;
	}
	bool valid = false;
	Variant result;
	evaluate(OP_NOT_EQUAL, *this, other, result, valid);
	return result.operator bool();
}

bool Variant::operator<(const Variant &other) const {
	if (get_type() != other.get_type()) {
		return get_type() < other.get_type();
	}
	bool valid = false;
	Variant result;
	evaluate(OP_LESS, *this, other, result, valid);
	return result.operator bool();
}

void Variant::callp(const StringName &method, const Variant **args, int argcount, Variant &r_ret, GDExtensionCallError &r_error) {
	internal::gdextension_interface_variant_call(_native_ptr(), method._native_ptr(), reinterpret_cast<GDExtensionConstVariantPtr *>(args), argcount, r_ret._native_ptr(), &r_error);
}

void Variant::callp_static(Variant::Type type, const StringName &method, const Variant **args, int argcount, Variant &r_ret, GDExtensionCallError &r_error) {
	internal::gdextension_interface_variant_call_static(static_cast<GDExtensionVariantType>(type), method._native_ptr(), reinterpret_cast<GDExtensionConstVariantPtr *>(args), argcount, r_ret._native_ptr(), &r_error);
}

void Variant::evaluate(const Operator &op, const Variant &a, const Variant &b, Variant &r_ret, bool &r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_evaluate(static_cast<GDExtensionVariantOperator>(op), a._native_ptr(), b._native_ptr(), r_ret._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
}

void Variant::set(const Variant &key, const Variant &value, bool *r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_set(_native_ptr(), key._native_ptr(), value._native_ptr(), &valid);
	if (r_valid) {
		*r_valid = PtrToArg<bool>::convert(&valid);
	}
}

void Variant::set_named(const StringName &name, const Variant &value, bool &r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_set_named(_native_ptr(), name._native_ptr(), value._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
}

void Variant::set_indexed(int64_t index, const Variant &value, bool &r_valid, bool &r_oob) {
	GDExtensionBool valid, oob;
	internal::gdextension_interface_variant_set_indexed(_native_ptr(), index, value._native_ptr(), &valid, &oob);
	r_valid = PtrToArg<bool>::convert(&valid);
	r_oob = PtrToArg<bool>::convert(&oob);
}

void Variant::set_keyed(const Variant &key, const Variant &value, bool &r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_set_keyed(_native_ptr(), key._native_ptr(), value._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
}

Variant Variant::get(const Variant &key, bool *r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_get(_native_ptr(), key._native_ptr(), result._native_ptr(), &valid);
	if (r_valid) {
		*r_valid = PtrToArg<bool>::convert(&valid);
	}
	return result;
}

Variant Variant::get_named(const StringName &name, bool &r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_get_named(_native_ptr(), name._native_ptr(), result._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return result;
}

Variant Variant::get_indexed(int64_t index, bool &r_valid, bool &r_oob) const {
	Variant result;
	GDExtensionBool valid;
	GDExtensionBool oob;
	internal::gdextension_interface_variant_get_indexed(_native_ptr(), index, result._native_ptr(), &valid, &oob);
	r_valid = PtrToArg<bool>::convert(&valid);
	r_oob = PtrToArg<bool>::convert(&oob);
	return result;
}

Variant Variant::get_keyed(const Variant &key, bool &r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_get_keyed(_native_ptr(), key._native_ptr(), result._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return result;
}

bool Variant::in(const Variant &index, bool *r_valid) const {
	Variant result;
	bool valid;
	evaluate(OP_IN, *this, index, result, valid);
	if (r_valid) {
		*r_valid = valid;
	}
	return result.operator bool();
}

bool Variant::iter_init(Variant &r_iter, bool &r_valid) const {
	GDExtensionBool valid;
	GDExtensionBool result = internal::gdextension_interface_variant_iter_init(_native_ptr(), r_iter._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return PtrToArg<bool>::convert(&result);
}

bool Variant::iter_next(Variant &r_iter, bool &r_valid) const {
	GDExtensionBool valid;
	GDExtensionBool result = internal::gdextension_interface_variant_iter_next(_native_ptr(), r_iter._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return PtrToArg<bool>::convert(&result);
}

Variant Variant::iter_get(const Variant &r_iter, bool &r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_iter_get(_native_ptr(), r_iter._native_ptr(), result._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return result;
}

Variant::Type Variant::get_type() const {
	return static_cast<Variant::Type>(internal::gdextension_interface_variant_get_type(_native_ptr()));
}

bool Variant::has_method(const StringName &method) const {
	GDExtensionBool has = internal::gdextension_interface_variant_has_method(_native_ptr(), method._native_ptr());
	return PtrToArg<bool>::convert(&has);
}

bool Variant::has_key(const Variant &key, bool *r_valid) const {
	GDExtensionBool valid;
	GDExtensionBool has = internal::gdextension_interface_variant_has_key(_native_ptr(), key._native_ptr(), &valid);
	if (r_valid) {
		*r_valid = PtrToArg<bool>::convert(&valid);
	}
	return PtrToArg<bool>::convert(&has);
}

bool Variant::has_member(Variant::Type type, const StringName &member) {
	GDExtensionBool has = internal::gdextension_interface_variant_has_member(static_cast<GDExtensionVariantType>(type), member._native_ptr());
	return PtrToArg<bool>::convert(&has);
}

uint32_t Variant::hash() const {
	GDExtensionInt hash = internal::gdextension_interface_variant_hash(_native_ptr());
	return PtrToArg<uint32_t>::convert(&hash);
}

uint32_t Variant::recursive_hash(int recursion_count) const {
	GDExtensionInt hash = internal::gdextension_interface_variant_recursive_hash(_native_ptr(), recursion_count);
	return PtrToArg<uint32_t>::convert(&hash);
}

bool Variant::hash_compare(const Variant &variant) const {
	GDExtensionBool compare = internal::gdextension_interface_variant_hash_compare(_native_ptr(), variant._native_ptr());
	return PtrToArg<bool>::convert(&compare);
}

bool Variant::booleanize() const {
	GDExtensionBool booleanized = internal::gdextension_interface_variant_booleanize(_native_ptr());
	return PtrToArg<bool>::convert(&booleanized);
}

PyObject *Variant::pythonize(const Dictionary &type_hints) const {
	// GIL must be active, caller is responsible
	PyObject *obj;

	switch (get_type()) {
		case Type::STRING:
		case Type::STRING_NAME:
		case Type::NODE_PATH:
		{
			String s = operator String();
			obj = s.py_str();
			break;
		}
		case Type::BOOL:
		{
			bool b = operator bool();
			obj = b ? Py_True : Py_False;
			Py_INCREF(obj);
			break;
		}
		case Type::INT:
		{
			int64_t i = operator int64_t();
			obj = PyLong_FromSsize_t(i);
			ERR_FAIL_NULL_V(obj, nullptr);
			break;
		}
		case Type::FLOAT:
		{
			double d = operator double();
			obj = PyFloat_FromDouble(d);
			ERR_FAIL_NULL_V(obj, nullptr);
			break;
		}
		case Type::VECTOR2:
		{
			Vector2 vec = operator Vector2();
			obj = PyStructSequence_New(&Vector2_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyFloat_FromDouble(vec.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(vec.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			break;
		}
		case Type::VECTOR2I:
		{
			Vector2i vec = operator Vector2i();
			obj = PyStructSequence_New(&Vector2i_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyLong_FromSsize_t(vec.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyLong_FromSsize_t(vec.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			break;
		}
		case Type::RECT2:
		{
			Rect2 rect = operator Rect2();
			obj = PyStructSequence_New(&Rect2_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *position = PyStructSequence_New(&Vector2_Type);
			ERR_FAIL_NULL_V(position, nullptr);
			PyObject *size = PyStructSequence_New(&Size2_Type);
			ERR_FAIL_NULL_V(size, nullptr);
			PyObject *x = PyFloat_FromDouble(rect.position.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(rect.position.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *width = PyFloat_FromDouble(rect.size.width);
			ERR_FAIL_NULL_V(width, nullptr);
			PyObject *height = PyFloat_FromDouble(rect.size.height);
			ERR_FAIL_NULL_V(height, nullptr);
			PyStructSequence_SET_ITEM(position, 0, x);
			PyStructSequence_SET_ITEM(position, 1, y);
			PyStructSequence_SET_ITEM(size, 0, width);
			PyStructSequence_SET_ITEM(size, 1, height);
			PyStructSequence_SET_ITEM(obj, 0, position);
			PyStructSequence_SET_ITEM(obj, 1, size);
			break;
		}
		case Type::RECT2I:
		{
			Rect2i rect = operator Rect2i();
			obj = PyStructSequence_New(&Rect2i_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *position = PyStructSequence_New(&Vector2i_Type);
			ERR_FAIL_NULL_V(position, nullptr);
			PyObject *size = PyStructSequence_New(&Size2_Type);
			ERR_FAIL_NULL_V(size, nullptr);
			PyObject *x = PyLong_FromSsize_t(rect.position.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyLong_FromSsize_t(rect.position.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *width = PyLong_FromSsize_t(rect.size.width);
			ERR_FAIL_NULL_V(width, nullptr);
			PyObject *height = PyLong_FromSsize_t(rect.size.height);
			ERR_FAIL_NULL_V(height, nullptr);
			PyStructSequence_SET_ITEM(position, 0, x);
			PyStructSequence_SET_ITEM(position, 1, y);
			PyStructSequence_SET_ITEM(size, 0, width);
			PyStructSequence_SET_ITEM(size, 1, height);
			PyStructSequence_SET_ITEM(obj, 0, position);
			PyStructSequence_SET_ITEM(obj, 1, size);
			break;
		}
		case Type::VECTOR3:
		{
			Vector3 vec = operator Vector3();
			obj = PyStructSequence_New(&Vector3_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyFloat_FromDouble(vec.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(vec.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyFloat_FromDouble(vec.z);
			ERR_FAIL_NULL_V(z, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			PyStructSequence_SET_ITEM(obj, 2, z);
			break;
		}
		case Type::VECTOR3I:
		{
			Vector3i vec = operator Vector3i();
			obj = PyStructSequence_New(&Vector3i_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyLong_FromSsize_t(vec.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyLong_FromSsize_t(vec.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyLong_FromSsize_t(vec.z);
			ERR_FAIL_NULL_V(z, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			PyStructSequence_SET_ITEM(obj, 2, z);
			break;
		}
		case Type::TRANSFORM2D:
		{
			Transform2D t = operator Transform2D();
			obj = PyTuple_New(3);
			ERR_FAIL_NULL_V(obj, nullptr);

			PyObject *x = PyTuple_New(2);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyTuple_New(2);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *o = PyTuple_New(2);
			ERR_FAIL_NULL_V(o, nullptr);

			PyObject *xx = PyFloat_FromDouble(t.columns[0][0]);
			ERR_FAIL_NULL_V(xx, nullptr);
			PyObject *xy = PyFloat_FromDouble(t.columns[0][1]);
			ERR_FAIL_NULL_V(xy, nullptr);
			PyTuple_SET_ITEM(x, 0, xx);
			PyTuple_SET_ITEM(x, 1, xy);
			PyObject *yx = PyFloat_FromDouble(t.columns[1][0]);
			ERR_FAIL_NULL_V(yx, nullptr);
			PyObject *yy = PyFloat_FromDouble(t.columns[1][1]);
			ERR_FAIL_NULL_V(yy, nullptr);
			PyTuple_SET_ITEM(y, 0, yx);
			PyTuple_SET_ITEM(y, 1, yy);
			PyObject *ox = PyFloat_FromDouble(t.columns[2][0]);
			ERR_FAIL_NULL_V(ox, nullptr);
			PyObject *oy = PyFloat_FromDouble(t.columns[2][1]);
			ERR_FAIL_NULL_V(oy, nullptr);
			PyTuple_SET_ITEM(o, 0, ox);
			PyTuple_SET_ITEM(o, 1, oy);

			PyTuple_SET_ITEM(obj, 0, x);
			PyTuple_SET_ITEM(obj, 1, y);
			PyTuple_SET_ITEM(obj, 2, o);
			break;
		}
		case Type::VECTOR4: {
			Vector4 vec = operator Vector4();
			obj = PyStructSequence_New(&Vector4_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyFloat_FromDouble(vec.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(vec.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyFloat_FromDouble(vec.z);
			ERR_FAIL_NULL_V(z, nullptr);
			PyObject *w = PyFloat_FromDouble(vec.w);
			ERR_FAIL_NULL_V(w, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			PyStructSequence_SET_ITEM(obj, 2, z);
			PyStructSequence_SET_ITEM(obj, 3, w);
			break;
		}
		case Type::VECTOR4I:
		{
			Vector4i vec = operator Vector4i();
			obj = PyStructSequence_New(&Vector4i_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyLong_FromSsize_t(vec.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyLong_FromSsize_t(vec.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyLong_FromSsize_t(vec.z);
			ERR_FAIL_NULL_V(z, nullptr);
			PyObject *w = PyLong_FromSsize_t(vec.w);
			ERR_FAIL_NULL_V(w, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			PyStructSequence_SET_ITEM(obj, 2, z);
			PyStructSequence_SET_ITEM(obj, 3, w);
			break;
		}
		case Type::PLANE:
		{
			Plane plane = operator Plane();
			obj = PyStructSequence_New(&Plane_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *normal = PyStructSequence_New(&Vector3_Type);
			ERR_FAIL_NULL_V(normal, nullptr);
			PyObject *d = PyFloat_FromDouble(plane.d);;
			ERR_FAIL_NULL_V(d, nullptr);
			PyObject *x = PyFloat_FromDouble(plane.normal.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(plane.normal.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyFloat_FromDouble(plane.normal.z);
			ERR_FAIL_NULL_V(z, nullptr);

			PyStructSequence_SET_ITEM(normal, 0, x);
			PyStructSequence_SET_ITEM(normal, 1, y);
			PyStructSequence_SET_ITEM(normal, 2, z);
			PyStructSequence_SET_ITEM(obj, 0, normal);
			PyStructSequence_SET_ITEM(obj, 1, d);
			break;
		}
		case Type::QUATERNION:
		{
			Quaternion q = operator Quaternion();
			obj = PyStructSequence_New(&Quaternion_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *x = PyFloat_FromDouble(q.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(q.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyFloat_FromDouble(q.z);
			ERR_FAIL_NULL_V(z, nullptr);
			PyObject *w = PyFloat_FromDouble(q.w);
			ERR_FAIL_NULL_V(w, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, x);
			PyStructSequence_SET_ITEM(obj, 1, y);
			PyStructSequence_SET_ITEM(obj, 2, z);
			PyStructSequence_SET_ITEM(obj, 3, w);
			break;
		}
		case Type::AABB:
		{
			godot::AABB aabb = operator godot::AABB();
			obj = PyStructSequence_New(&AABB_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *position = PyStructSequence_New(&Vector3_Type);
			ERR_FAIL_NULL_V(position, nullptr);
			PyObject *size = PyStructSequence_New(&Vector3_Type);
			ERR_FAIL_NULL_V(size, nullptr);
			PyObject *x = PyFloat_FromDouble(aabb.position.x);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyFloat_FromDouble(aabb.position.y);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyFloat_FromDouble(aabb.position.z);
			ERR_FAIL_NULL_V(z, nullptr);
			PyObject *sx = PyFloat_FromDouble(aabb.size.x);
			ERR_FAIL_NULL_V(sx, nullptr);
			PyObject *sy = PyFloat_FromDouble(aabb.size.y);
			ERR_FAIL_NULL_V(sy, nullptr);
			PyObject *sz = PyFloat_FromDouble(aabb.size.z);
			ERR_FAIL_NULL_V(sz, nullptr);
			PyStructSequence_SET_ITEM(position, 0, x);
			PyStructSequence_SET_ITEM(position, 1, y);
			PyStructSequence_SET_ITEM(position, 2, z);
			PyStructSequence_SET_ITEM(size, 0, sx);
			PyStructSequence_SET_ITEM(size, 1, sy);
			PyStructSequence_SET_ITEM(size, 2, sz);
			PyStructSequence_SET_ITEM(obj, 0, position);
			PyStructSequence_SET_ITEM(obj, 1, size);
			break;
		}
		case Type::BASIS:
		{
			Basis b = operator Basis();
			obj = PyTuple_New(3);
			ERR_FAIL_NULL_V(obj, nullptr);

			PyObject *x = PyTuple_New(3);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyTuple_New(3);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyTuple_New(3);
			ERR_FAIL_NULL_V(z, nullptr);

			PyObject *xx = PyFloat_FromDouble(b.rows[0][0]);
			ERR_FAIL_NULL_V(xx, nullptr);
			PyObject *xy = PyFloat_FromDouble(b.rows[0][1]);
			ERR_FAIL_NULL_V(xy, nullptr);
			PyObject *xz = PyFloat_FromDouble(b.rows[0][2]);
			ERR_FAIL_NULL_V(xz, nullptr);
			PyTuple_SET_ITEM(x, 0, xx);
			PyTuple_SET_ITEM(x, 1, xy);
			PyTuple_SET_ITEM(x, 2, xz);
			PyObject *yx = PyFloat_FromDouble(b.rows[1][0]);
			ERR_FAIL_NULL_V(yx, nullptr);
			PyObject *yy = PyFloat_FromDouble(b.rows[1][1]);
			ERR_FAIL_NULL_V(yy, nullptr);
			PyObject *yz = PyFloat_FromDouble(b.rows[1][2]);
			ERR_FAIL_NULL_V(yz, nullptr);
			PyTuple_SET_ITEM(y, 0, yx);
			PyTuple_SET_ITEM(y, 1, yy);
			PyTuple_SET_ITEM(y, 2, yz);
			PyObject *zx = PyFloat_FromDouble(b.rows[2][0]);
			ERR_FAIL_NULL_V(zx, nullptr);
			PyObject *zy = PyFloat_FromDouble(b.rows[2][1]);
			ERR_FAIL_NULL_V(zy, nullptr);
			PyObject *zz = PyFloat_FromDouble(b.rows[2][2]);
			ERR_FAIL_NULL_V(zz, nullptr);
			PyTuple_SET_ITEM(z, 0, zx);
			PyTuple_SET_ITEM(z, 1, zy);
			PyTuple_SET_ITEM(z, 2, zz);

			PyTuple_SET_ITEM(obj, 0, x);
			PyTuple_SET_ITEM(obj, 1, y);
			PyTuple_SET_ITEM(obj, 2, z);
			break;
		}
		case Type::TRANSFORM3D:
		{
			Transform3D t = operator Transform3D();
			obj = PyStructSequence_New(&Transform3D_Type);
			ERR_FAIL_NULL_V(obj, nullptr);

			PyObject *basis = PyTuple_New(3);
			ERR_FAIL_NULL_V(obj, nullptr);
			Basis b = operator Basis();

			PyObject *x = PyTuple_New(3);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyTuple_New(3);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyTuple_New(3);
			ERR_FAIL_NULL_V(z, nullptr);

			PyObject *xx = PyFloat_FromDouble(t.basis.rows[0][0]);
			ERR_FAIL_NULL_V(xx, nullptr);
			PyObject *xy = PyFloat_FromDouble(t.basis.rows[0][1]);
			ERR_FAIL_NULL_V(xy, nullptr);
			PyObject *xz = PyFloat_FromDouble(t.basis.rows[0][2]);
			ERR_FAIL_NULL_V(xz, nullptr);
			PyTuple_SET_ITEM(x, 0, xx);
			PyTuple_SET_ITEM(x, 1, xy);
			PyTuple_SET_ITEM(x, 2, xz);
			PyObject *yx = PyFloat_FromDouble(t.basis.rows[1][0]);
			ERR_FAIL_NULL_V(yx, nullptr);
			PyObject *yy = PyFloat_FromDouble(t.basis.rows[1][1]);
			ERR_FAIL_NULL_V(yy, nullptr);
			PyObject *yz = PyFloat_FromDouble(t.basis.rows[1][2]);
			ERR_FAIL_NULL_V(yz, nullptr);
			PyTuple_SET_ITEM(y, 0, yx);
			PyTuple_SET_ITEM(y, 1, yy);
			PyTuple_SET_ITEM(y, 2, yz);
			PyObject *zx = PyFloat_FromDouble(t.basis.rows[2][0]);
			ERR_FAIL_NULL_V(zx, nullptr);
			PyObject *zy = PyFloat_FromDouble(t.basis.rows[2][1]);
			ERR_FAIL_NULL_V(zy, nullptr);
			PyObject *zz = PyFloat_FromDouble(t.basis.rows[2][2]);
			ERR_FAIL_NULL_V(zz, nullptr);
			PyTuple_SET_ITEM(z, 0, zx);
			PyTuple_SET_ITEM(z, 1, zy);
			PyTuple_SET_ITEM(z, 2, zz);

			PyTuple_SET_ITEM(basis, 0, x);
			PyTuple_SET_ITEM(basis, 1, y);
			PyTuple_SET_ITEM(basis, 2, z);

			PyObject *origin = PyStructSequence_New(&Vector3_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			Vector3 vec = operator Vector3();
			PyObject *ox = PyFloat_FromDouble(t.origin.x);
			ERR_FAIL_NULL_V(ox, nullptr);
			PyObject *oy = PyFloat_FromDouble(t.origin.y);
			ERR_FAIL_NULL_V(oy, nullptr);
			PyObject *oz = PyFloat_FromDouble(t.origin.z);
			ERR_FAIL_NULL_V(oz, nullptr);
			PyStructSequence_SET_ITEM(origin, 0, ox);
			PyStructSequence_SET_ITEM(origin, 1, oy);
			PyStructSequence_SET_ITEM(origin, 2, oz);

			PyStructSequence_SET_ITEM(obj, 0, basis);
			PyStructSequence_SET_ITEM(obj, 1, origin);
		}
		case Type::PROJECTION:
		{
			Projection p = operator Projection();
			obj = PyTuple_New(4);
			ERR_FAIL_NULL_V(obj, nullptr);

			PyObject *x = PyTuple_New(4);
			ERR_FAIL_NULL_V(x, nullptr);
			PyObject *y = PyTuple_New(4);
			ERR_FAIL_NULL_V(y, nullptr);
			PyObject *z = PyTuple_New(4);
			ERR_FAIL_NULL_V(z, nullptr);
			PyObject *w = PyTuple_New(4);
			ERR_FAIL_NULL_V(w, nullptr);

			PyObject *xx = PyFloat_FromDouble(p.columns[0][0]);
			ERR_FAIL_NULL_V(xx, nullptr);
			PyObject *xy = PyFloat_FromDouble(p.columns[0][1]);
			ERR_FAIL_NULL_V(xy, nullptr);
			PyObject *xz = PyFloat_FromDouble(p.columns[0][2]);
			ERR_FAIL_NULL_V(xz, nullptr);
			PyObject *xw = PyFloat_FromDouble(p.columns[0][3]);
			ERR_FAIL_NULL_V(xw, nullptr);
			PyTuple_SET_ITEM(x, 0, xx);
			PyTuple_SET_ITEM(x, 1, xy);
			PyTuple_SET_ITEM(x, 2, xz);
			PyTuple_SET_ITEM(x, 3, xw);
			PyObject *yx = PyFloat_FromDouble(p.columns[1][0]);
			ERR_FAIL_NULL_V(yx, nullptr);
			PyObject *yy = PyFloat_FromDouble(p.columns[1][1]);
			ERR_FAIL_NULL_V(yy, nullptr);
			PyObject *yz = PyFloat_FromDouble(p.columns[1][2]);
			ERR_FAIL_NULL_V(yz, nullptr);
			PyObject *yw = PyFloat_FromDouble(p.columns[1][3]);
			ERR_FAIL_NULL_V(yw, nullptr);
			PyTuple_SET_ITEM(y, 0, yx);
			PyTuple_SET_ITEM(y, 1, yy);
			PyTuple_SET_ITEM(y, 2, yz);
			PyTuple_SET_ITEM(y, 3, yw);
			PyObject *zx = PyFloat_FromDouble(p.columns[2][0]);
			ERR_FAIL_NULL_V(zx, nullptr);
			PyObject *zy = PyFloat_FromDouble(p.columns[2][1]);
			ERR_FAIL_NULL_V(zy, nullptr);
			PyObject *zz = PyFloat_FromDouble(p.columns[2][2]);
			ERR_FAIL_NULL_V(zz, nullptr);
			PyObject *zw = PyFloat_FromDouble(p.columns[2][3]);
			ERR_FAIL_NULL_V(zw, nullptr);
			PyTuple_SET_ITEM(z, 0, zx);
			PyTuple_SET_ITEM(z, 1, zy);
			PyTuple_SET_ITEM(z, 2, zz);
			PyTuple_SET_ITEM(z, 3, zw);
			PyObject *wx = PyFloat_FromDouble(p.columns[3][0]);
			ERR_FAIL_NULL_V(zx, nullptr);
			PyObject *wy = PyFloat_FromDouble(p.columns[3][1]);
			ERR_FAIL_NULL_V(zy, nullptr);
			PyObject *wz = PyFloat_FromDouble(p.columns[3][2]);
			ERR_FAIL_NULL_V(zz, nullptr);
			PyObject *ww = PyFloat_FromDouble(p.columns[3][3]);
			ERR_FAIL_NULL_V(zw, nullptr);
			PyTuple_SET_ITEM(w, 0, wx);
			PyTuple_SET_ITEM(w, 1, wy);
			PyTuple_SET_ITEM(w, 2, wz);
			PyTuple_SET_ITEM(w, 3, ww);

			PyTuple_SET_ITEM(obj, 0, x);
			PyTuple_SET_ITEM(obj, 1, y);
			PyTuple_SET_ITEM(obj, 2, z);
			PyTuple_SET_ITEM(obj, 3, z);
			break;
		}
		case Type::COLOR:
		{
			Color c = operator Color();
			obj = PyStructSequence_New(&Color_Type);
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *r = PyFloat_FromDouble(static_cast<double>(c.r));
			ERR_FAIL_NULL_V(r, nullptr);
			PyObject *g = PyFloat_FromDouble(static_cast<double>(c.g));
			ERR_FAIL_NULL_V(g, nullptr);
			PyObject *b = PyFloat_FromDouble(static_cast<double>(c.b));
			ERR_FAIL_NULL_V(b, nullptr);
			PyObject *a = PyFloat_FromDouble(static_cast<double>(c.a));
			ERR_FAIL_NULL_V(a, nullptr);
			PyStructSequence_SET_ITEM(obj, 0, r);
			PyStructSequence_SET_ITEM(obj, 1, g);
			PyStructSequence_SET_ITEM(obj, 2, b);
			PyStructSequence_SET_ITEM(obj, 3, a);
			break;
		}
		case Type::RID:
		{
			godot::RID rid = operator godot::RID();
			obj =  PyLong_FromSsize_t(int64_t(rid.get_id()));
			ERR_FAIL_NULL_V(obj, nullptr);
			break;
		}
		case Type::CALLABLE:
		case Type::SIGNAL:
		{
			Py_INCREF(Py_None);
			obj = Py_None;
			ERR_PRINT("NOT IMPLEMENTED: PyObject* from Callable/Signal Variants. Returning None");
			break;
		}
		case Type::OBJECT:
		{
			Object *o = operator Object *();
			obj = _get_object_from_owner(o->_owner, o->get_class());
			break;
		}
		case Type::DICTIONARY:
		{
			const Dictionary dict = operator Dictionary();
			const Array keys = dict.keys();
			obj = PyDict_New();
			ERR_FAIL_NULL_V(obj, nullptr);

			for (size_t i = 0; i < keys.size(); i++) {
				Variant _key = keys[i];
				PyObject *key = _key;
				ERR_FAIL_NULL_V(key, nullptr);
				PyObject *val = dict[_key];
				ERR_FAIL_NULL_V(val, nullptr);
				PyDict_SetItem(obj, key, val);
			}
			break;
		}
		case Type::ARRAY:
		{
			const Array arr = operator Array();
			if (type_hints.has("Array") && type_hints["Array"] == "tuple") {
				obj = PyTuple_New(arr.size());
				ERR_FAIL_NULL_V(obj, nullptr);

				for (size_t i = 0; i < arr.size(); i++) {
					PyObject *item = arr[i];
					ERR_FAIL_NULL_V(item, nullptr);
					PyTuple_SET_ITEM(obj, i, item);
				}
			} else {
				obj = PyList_New(arr.size());
				ERR_FAIL_NULL_V(obj, nullptr);

				for (size_t i = 0; i < arr.size(); i++) {
					PyObject *item = arr[i];
					ERR_FAIL_NULL_V(item, nullptr);
					PyList_SET_ITEM(obj, i, item);
				}
			}
			break;
		}

		// TODO: Return NumPy arrays as an option, maybe make NumPy default for
		//       all numeric arrays

		case Variant::Type::PACKED_BYTE_ARRAY:
		{
			PackedByteArray a = operator PackedByteArray();
			if (type_hints.has("PackedByteArray") && type_hints["PackedByteArray"] == "bytes") {
				obj = PyBytes_FromStringAndSize(reinterpret_cast<const char *>(a.ptr()), a.size());
			} else {
				obj = PyByteArray_FromStringAndSize(reinterpret_cast<const char *>(a.ptr()), a.size());
			}
			ERR_FAIL_NULL_V(obj, nullptr);
			break;
		}
		case Variant::Type::PACKED_INT32_ARRAY:
		{
			PackedInt32Array a = operator PackedInt32Array();
			obj = PyTuple_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			for (size_t i = 0; i < a.size(); i++) {
				PyObject *elem = PyLong_FromSsize_t(int64_t(a[i]));
				ERR_FAIL_NULL_V(elem, nullptr);
				PyTuple_SET_ITEM(obj, i, nullptr);
			}
			break;
		}
		case Variant::Type::PACKED_INT64_ARRAY:
		{
			PackedInt64Array a = operator PackedInt64Array();
			obj = PyTuple_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			for (size_t i = 0; i < a.size(); i++) {
				PyObject *elem = PyLong_FromSsize_t(a[i]);
				ERR_FAIL_NULL_V(elem, nullptr);
				PyTuple_SET_ITEM(obj, i, elem);
			}
			break;
		}
		case Variant::Type::PACKED_FLOAT32_ARRAY:
		{
			PackedFloat32Array a = operator PackedFloat32Array();
			obj = PyTuple_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			for (size_t i = 0; i < a.size(); i++) {
				PyObject *elem = PyFloat_FromDouble(static_cast<double>(a[i]));
				ERR_FAIL_NULL_V(elem, nullptr);
				PyTuple_SET_ITEM(obj, i, elem);
			}
			break;
		}
		case Variant::Type::PACKED_FLOAT64_ARRAY:
		{
			PackedFloat64Array a = operator PackedFloat64Array();
			obj = PyTuple_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			for (size_t i = 0; i < a.size(); i++) {
				PyObject *elem = PyFloat_FromDouble(a[i]);
				ERR_FAIL_NULL_V(elem, nullptr);
				PyTuple_SET_ITEM(obj, i, elem);
			}
			break;
		}
		case Variant::Type::PACKED_STRING_ARRAY:
		{
			PackedStringArray a = operator PackedStringArray();
			obj = PyTuple_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			for (size_t i = 0; i < a.size(); i++) {
				PyObject *elem = a[i].py_str();
				ERR_FAIL_NULL_V(elem, nullptr);
				PyTuple_SET_ITEM(obj, i, elem);
			}
			break;
		}
		case Variant::Type::PACKED_VECTOR2_ARRAY:
		{
			PackedVector2Array a = operator PackedVector2Array();
			obj = PyList_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *vec;
			for (size_t i = 0; i < a.size(); i++) {
				vec = PyStructSequence_New(&Vector2_Type);
				ERR_FAIL_NULL_V(vec, nullptr);
				Vector2 elem = a[i];
				PyObject *x = PyFloat_FromDouble(static_cast<double>(elem.x));
				ERR_FAIL_NULL_V(x, nullptr);
				PyObject *y = PyFloat_FromDouble(static_cast<double>(elem.y));
				ERR_FAIL_NULL_V(y, nullptr);
				PyStructSequence_SET_ITEM(vec, 0, x);
				PyStructSequence_SET_ITEM(vec, 1, y);
				PyList_SET_ITEM(obj, i, vec);
			}
			break;
		}
		case Type::PACKED_VECTOR3_ARRAY:
		{
			PackedVector3Array a = operator PackedVector3Array();
			obj = PyList_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *vec;
			for (size_t i = 0; i < a.size(); i++) {
				vec = PyStructSequence_New(&Vector3_Type);
				ERR_FAIL_NULL_V(vec, nullptr);
				Vector3 elem = a[i];
				PyObject *x = PyFloat_FromDouble(static_cast<double>(elem.x));
				ERR_FAIL_NULL_V(x, nullptr);
				PyObject *y = PyFloat_FromDouble(static_cast<double>(elem.y));
				ERR_FAIL_NULL_V(y, nullptr);
				PyObject *z = PyFloat_FromDouble(static_cast<double>(elem.z));
				ERR_FAIL_NULL_V(z, nullptr);
				PyStructSequence_SET_ITEM(vec, 0, x);
				PyStructSequence_SET_ITEM(vec, 1, y);
				PyStructSequence_SET_ITEM(vec, 2, z);
				PyList_SET_ITEM(obj, i, vec);
			}
			break;
		}
		case Type::PACKED_COLOR_ARRAY:
		{
			PackedColorArray a = operator PackedColorArray();
			obj = PyList_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *c;
			for (size_t i = 0; i < a.size(); i++) {
				c = PyStructSequence_New(&Color_Type);
				ERR_FAIL_NULL_V(c, nullptr);
				Color elem = a[i];
				PyObject *r = PyFloat_FromDouble(static_cast<double>(elem.r));
				ERR_FAIL_NULL_V(r, nullptr);
				PyObject *g = PyFloat_FromDouble(static_cast<double>(elem.g));
				ERR_FAIL_NULL_V(g, nullptr);
				PyObject *b = PyFloat_FromDouble(static_cast<double>(elem.b));
				ERR_FAIL_NULL_V(b, nullptr);
				PyObject *a = PyFloat_FromDouble(static_cast<double>(elem.a));
				ERR_FAIL_NULL_V(a, nullptr);
				PyStructSequence_SET_ITEM(c, 0, r);
				PyStructSequence_SET_ITEM(c, 1, g);
				PyStructSequence_SET_ITEM(c, 2, b);
				PyStructSequence_SET_ITEM(c, 3, a);
				PyList_SET_ITEM(obj, i, c);
			}
			break;
		}
		case Type::PACKED_VECTOR4_ARRAY:
		{
			PackedVector4Array a = operator PackedVector4Array();
			obj = PyList_New(a.size());
			ERR_FAIL_NULL_V(obj, nullptr);
			PyObject *vec;
			for (size_t i = 0; i < a.size(); i++) {
				vec = PyStructSequence_New(&Vector4_Type);
				ERR_FAIL_NULL_V(vec, nullptr);
				Vector4 elem = a[i];
				PyObject *x = PyFloat_FromDouble(static_cast<double>(elem.x));
				ERR_FAIL_NULL_V(x, nullptr);
				PyObject *y = PyFloat_FromDouble(static_cast<double>(elem.y));
				ERR_FAIL_NULL_V(y, nullptr);
				PyObject *z = PyFloat_FromDouble(static_cast<double>(elem.z));
				ERR_FAIL_NULL_V(z, nullptr);
				PyObject *w = PyFloat_FromDouble(static_cast<double>(elem.w));
				ERR_FAIL_NULL_V(w, nullptr);
				PyStructSequence_SET_ITEM(vec, 0, x);
				PyStructSequence_SET_ITEM(vec, 1, y);
				PyStructSequence_SET_ITEM(vec, 2, z);
				PyStructSequence_SET_ITEM(vec, 3, w);
				PyList_SET_ITEM(obj, i, vec);
			}
			break;
		}
		case Variant::Type::NIL:
		{
			Py_INCREF(Py_None);
			obj = Py_None;
			break;
		}
		default:
			Py_INCREF(Py_None);
			obj = Py_None;
			ERR_PRINT(vformat("Unknown variant type %d, returning None", (int)get_type()));
			break;
	}

	return obj;
}

String Variant::stringify() const {
	String result;
	internal::gdextension_interface_variant_stringify(_native_ptr(), result._native_ptr());
	return result;
}

Variant Variant::duplicate(bool deep) const {
	Variant result;
	GDExtensionBool _deep;
	PtrToArg<bool>::encode(deep, &_deep);
	internal::gdextension_interface_variant_duplicate(_native_ptr(), result._native_ptr(), _deep);
	return result;
}

String Variant::get_type_name(Variant::Type type) {
	String result;
	internal::gdextension_interface_variant_get_type_name(static_cast<GDExtensionVariantType>(type), result._native_ptr());
	return result;
}

bool Variant::can_convert(Variant::Type from, Variant::Type to) {
	GDExtensionBool can = internal::gdextension_interface_variant_can_convert(static_cast<GDExtensionVariantType>(from), static_cast<GDExtensionVariantType>(to));
	return PtrToArg<bool>::convert(&can);
}

bool Variant::can_convert_strict(Variant::Type from, Variant::Type to) {
	GDExtensionBool can = internal::gdextension_interface_variant_can_convert_strict(static_cast<GDExtensionVariantType>(from), static_cast<GDExtensionVariantType>(to));
	return PtrToArg<bool>::convert(&can);
}

void Variant::clear() {
	static const bool needs_deinit[Variant::VARIANT_MAX] = {
		false, // NIL,
		false, // BOOL,
		false, // INT,
		false, // FLOAT,
		true, // STRING,
		false, // VECTOR2,
		false, // VECTOR2I,
		false, // RECT2,
		false, // RECT2I,
		false, // VECTOR3,
		false, // VECTOR3I,
		true, // TRANSFORM2D,
		false, // VECTOR4,
		false, // VECTOR4I,
		false, // PLANE,
		false, // QUATERNION,
		true, // AABB,
		true, // BASIS,
		true, // TRANSFORM3D,
		true, // PROJECTION,

		// misc types
		false, // COLOR,
		true, // STRING_NAME,
		true, // NODE_PATH,
		false, // RID,
		true, // OBJECT,
		true, // CALLABLE,
		true, // SIGNAL,
		true, // DICTIONARY,
		true, // ARRAY,

		// typed arrays
		true, // PACKED_BYTE_ARRAY,
		true, // PACKED_INT32_ARRAY,
		true, // PACKED_INT64_ARRAY,
		true, // PACKED_FLOAT32_ARRAY,
		true, // PACKED_FLOAT64_ARRAY,
		true, // PACKED_STRING_ARRAY,
		true, // PACKED_VECTOR2_ARRAY,
		true, // PACKED_VECTOR3_ARRAY,
		true, // PACKED_COLOR_ARRAY,
	};

	if (unlikely(needs_deinit[get_type()])) { // Make it fast for types that don't need deinit.
		internal::gdextension_interface_variant_destroy(_native_ptr());
	}
	internal::gdextension_interface_variant_new_nil(_native_ptr());
}

} // namespace godot
