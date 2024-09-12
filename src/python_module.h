#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

using namespace godot;

class PythonModule : public Resource {
	GDCLASS(PythonModule, Resource);

private:
    PyObject *obj;
    String name;

protected:
	static void _bind_methods();

public:
    PythonModule();
    ~PythonModule();

    static PythonModule *import_module(const String &name);
};
