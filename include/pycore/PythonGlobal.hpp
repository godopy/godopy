#ifndef PYTHON_GLOBAL_HPP
#define PYTHON_GLOBAL_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL PYGODOT_ARRAY_API
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API 0x0000000D
#endif
#include <numpy/arrayobject.h>

#include <gdnative_api_struct.gen.h>

namespace godopy {

// extern "C" const void **array_api;

class GodoPy {

public:
	static void python_preconfig(godot_gdnative_init_options *o);
	static void python_init();
	static void python_terminate();

	static void nativescript_init(void *handle, bool init_cython=true, bool init_python=true);
	static void nativescript_terminate(void *handle, bool terminate_cython=true, bool terminate_python=true);

	static void set_cython_language_index(int language_index);
	static void set_python_language_index(int language_index);
};

} // namespace godopy

#endif
