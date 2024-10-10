#pragma once

#include <godot_cpp/core/class_db.hpp>

#include "python_object.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

using namespace godot;

class PythonRuntime {
private:
	bool initialized;
	String library_path;

protected:
	static PythonRuntime *singleton;

public:
	static PythonRuntime *get_singleton() { return singleton; }

private:
	int set_config_paths(PyConfig *config);
	void pre_initialize();

	_ALWAYS_INLINE_ bool is_initialized() const { return initialized; }

public:
	void initialize();
	void finalize();

	void run_simple_string(const String &p_string_script);
	Ref<PythonObject> import_module(const String &p_name);

	PythonObject *python_object_from_pyobject(PyObject *);

	PythonRuntime();
	~PythonRuntime();
};
