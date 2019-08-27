#include "Dictionary.hpp"
#include "Array.hpp"
#include "GodotGlobal.hpp"
#include "Variant.hpp"

#include <stdexcept>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <internal-packages/godot/core/types.hpp>

namespace godot {

Dictionary::Dictionary() {
	godot::api->godot_dictionary_new(&_godot_dictionary);
}

Dictionary::Dictionary(const Dictionary &other) {
	godot::api->godot_dictionary_new_copy(&_godot_dictionary, &other._godot_dictionary);
}

Dictionary &Dictionary::operator=(const Dictionary &other) {
	godot::api->godot_dictionary_destroy(&_godot_dictionary);
	godot::api->godot_dictionary_new_copy(&_godot_dictionary, &other._godot_dictionary);
	return *this;
}

void Dictionary::clear() {
	godot::api->godot_dictionary_clear(&_godot_dictionary);
}

bool Dictionary::empty() const {
	return godot::api->godot_dictionary_empty(&_godot_dictionary);
}

void Dictionary::erase(const Variant &key) {
	godot::api->godot_dictionary_erase(&_godot_dictionary, (godot_variant *)&key);
}

bool Dictionary::has(const Variant &key) const {
	return godot::api->godot_dictionary_has(&_godot_dictionary, (godot_variant *)&key);
}

bool Dictionary::has_all(const Array &keys) const {
	return godot::api->godot_dictionary_has_all(&_godot_dictionary, (godot_array *)&keys);
}

uint32_t Dictionary::hash() const {
	return godot::api->godot_dictionary_hash(&_godot_dictionary);
}

Array Dictionary::keys() const {
	godot_array a = godot::api->godot_dictionary_keys(&_godot_dictionary);
	return *(Array *)&a;
}

Variant &Dictionary::operator[](const Variant &key) {
	return *(Variant *)godot::api->godot_dictionary_operator_index(&_godot_dictionary, (godot_variant *)&key);
}

const Variant &Dictionary::operator[](const Variant &key) const {
	// oops I did it again
	return *(Variant *)godot::api->godot_dictionary_operator_index((godot_dictionary *)&_godot_dictionary, (godot_variant *)&key);
}

int Dictionary::size() const {
	return godot::api->godot_dictionary_size(&_godot_dictionary);
}

String Dictionary::to_json() const {
	godot_string s = godot::api->godot_dictionary_to_json(&_godot_dictionary);
	return *(String *)&s;
}

Array Dictionary::values() const {
	godot_array a = godot::api->godot_dictionary_values(&_godot_dictionary);
	return *(Array *)&a;
}

Dictionary::~Dictionary() {
	godot::api->godot_dictionary_destroy(&_godot_dictionary);
}


Dictionary::Dictionary(const PyObject *other) {
	if (Py_TYPE(other) == PyGodotWrapperType_Dictionary) {
		godot_dictionary *p = _python_wrapper_to_godot_dictionary((PyObject *)other);

		if (likely(p)) {
			godot::api->godot_dictionary_new_copy(&_godot_dictionary, p);
		} else {
			// raises ValueError in Cython/Python context
			throw std::invalid_argument("invalid Python argument");
		}
	} else {
		// Not a C++ wrapper, initialize from the mapping protocol
		if (!PyMapping_Check((PyObject *)other)) {
			throw std::invalid_argument("invalid Python argument");
		}

		godot::api->godot_dictionary_new(&_godot_dictionary);
		PyObject *keys = PyMapping_Keys((PyObject *)other);
		// TODO: Add NULL checks
		for (int i = 0; i < PyMapping_Size((PyObject *)other); i++) {
			PyObject *key = PyList_GET_ITEM(keys, i);
			Variant _key = key;
			Variant _item = *(Variant *)godot::api->godot_dictionary_operator_index(&_godot_dictionary, (godot_variant *)&_key);
			_item = PyObject_GetItem((PyObject *)other, key);
		}
	}
}

PyObject *Dictionary::py_dict() const {
	PyObject *obj = PyDict_New();
	const Array _keys = keys();

	for (int i = 0; i < _keys.size(); i++) {
		Variant _key = _keys[i];
		Variant _val = *(Variant *)godot::api->godot_dictionary_operator_index((godot_dictionary *)&_godot_dictionary, (godot_variant *)&_key);
		PyObject *key = _key;
		PyObject *val = _val;
		// TODO: Check NULL pointers
		PyDict_SetItem(obj, key, val);
	}

	Py_INCREF(obj);
	return obj;
}

} // namespace godot
