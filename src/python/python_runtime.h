#pragma once

#include <godot_cpp/core/class_db.hpp>

#include "python_object.h"

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>

using namespace godot;

class PythonRuntime {
private:
	PyInterpreterState *interpreter_state;
	PyThreadState *main_thread_state;
	uint64_t main_thread_id;

	bool initialized;
	String library_path;

	std::unordered_map<uint64_t, PyThreadState *> thread_states;

protected:
	static PythonRuntime *singleton;

public:
	static PythonRuntime *get_singleton() { return singleton; }
	PyInterpreterState *get_interpreter_state() { return interpreter_state; }

private:
	int set_config_paths(PyConfig *config);
	void pre_initialize();

	_ALWAYS_INLINE_ bool is_initialized() const { return initialized; }

public:
	void initialize();
	void finalize();

	void run_simple_string(const String &p_string_script);
	Ref<PythonObject> import_module(const String &p_name);
	void init_module(const String &p_name);

	PythonObject *python_object_from_pyobject(PyObject *);

	void ensure_current_thread_state(bool setdefault=false);

	PythonRuntime();
	~PythonRuntime();
};
