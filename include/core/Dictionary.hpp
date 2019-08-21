#ifndef DICTIONARY_H
#define DICTIONARY_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "Variant.hpp"

#include "Array.hpp"

#include <gdnative/dictionary.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
// typedef struct __pyx_obj_5godot_10core_types_GodotDictionary *_python_dictionary_wrapper;

namespace godot {

class Dictionary {
	godot_dictionary _godot_dictionary;

public:
	Dictionary();
	Dictionary(const Dictionary &other);
	Dictionary &operator=(const Dictionary &other);

	template <class... Args>
	static Dictionary make(Args... args) {
		return helpers::add_all(Dictionary(), args...);
	}

	void clear();

	bool empty() const;

	void erase(const Variant &key);

	bool has(const Variant &key) const;

	bool has_all(const Array &keys) const;

	uint32_t hash() const;

	Array keys() const;

	Variant &operator[](const Variant &key);

	const Variant &operator[](const Variant &key) const;

	int size() const;

	String to_json() const;

	Array values() const;

	~Dictionary();

	Dictionary(const PyObject *other);

	PyObject *to_python_wrapper();
	PyObject *to_python();
};

} // namespace godot

#endif // DICTIONARY_H
