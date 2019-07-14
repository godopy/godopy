#ifndef PYTHON_GLOBAL_HPP
#define PYTHON_GLOBAL_HPP

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <gdnative_api_struct.gen.h>

namespace godot {

class Python {
public:
	static void set_pythonpath(godot_gdnative_init_options *o);
	static void init();
	static void terminate();
};


} // namespace godot

#endif
