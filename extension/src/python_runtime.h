#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

class PythonRuntime {
private:
	bool initialized;
	bool i_am_py;

protected:
	static PythonRuntime *singleton;

public:
	static PythonRuntime *get_singleton() { return singleton; }

	_FORCE_INLINE_ bool is_initialized() const { return initialized; }
	_FORCE_INLINE_ bool am_i_py() const { return i_am_py; }

	void pre_initialize();
	void initialize();
	bool detect_python_mode();
	void pre_initialize_as_py();
	void initialize_as_py();

	void run_simple_string(const String &p_string_script);
	int run_python_main();

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

	void initialize();

	void run_simple_string(const String &p_string_script);
	int run_main();

	Python();
	~Python();
};
