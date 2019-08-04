#ifndef NODEPATH_H
#define NODEPATH_H

#include "String.hpp"

#include <gdnative/node_path.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
// typedef struct __pyx_obj_5godot_10core_types_GodotNodePath *_python_nodepath_wrapper;

namespace godot {

class NodePath {
	godot_node_path _node_path;

public:
	NodePath();

	NodePath(const NodePath &other);

	NodePath(const String &from);

	NodePath(const char *contents);

	String get_name(const int idx) const;

	int get_name_count() const;

	String get_subname(const int idx) const;

	int get_subname_count() const;

	bool is_absolute() const;

	bool is_empty() const;

	NodePath get_as_property_path() const;

	String get_concatenated_subnames() const;

	PyObject *pythonize();

	operator String() const;

	void operator=(const NodePath &other);

	bool operator==(const NodePath &other);

	~NodePath();
};

} // namespace godot

#endif // NODEPATH_H
