#pragma once

#include <godot_cpp/core/class_db.hpp>

#include "python_object.h"

using namespace godot;

class PythonRuntime {
private:
	bool initialized;

protected:
	static PythonRuntime *singleton;

public:
	static PythonRuntime *get_singleton() { return singleton; }

	_ALWAYS_INLINE_ bool is_initialized() const { return initialized; }

	void pre_initialize();
	void initialize();

	void run_simple_string(const String &p_string_script);
	PythonObject *import_module(const String &p_name);

	PythonRuntime();
	~PythonRuntime();
};

class Python : public Object {
	GDCLASS(Python, Object);

	friend class PythonRuntime;

	static Python *singleton;

protected:
	static void _bind_methods();

public:
	static Python *get_singleton() { return singleton; }

	void run_simple_string(const String &p_string);
	PythonObject *import_module(const String &p_name);

	Python();
	~Python();
};