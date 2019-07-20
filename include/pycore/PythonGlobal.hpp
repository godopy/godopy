#ifndef PYTHON_GLOBAL_HPP
#define PYTHON_GLOBAL_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <gdnative_api_struct.gen.h>

PyMODINIT_FUNC PyInit__pygodot(void);

// Cython ignores C++ namespaces for Python types
typedef struct {
  PyObject_HEAD
  godot_object *_owner;
  size_t _type_tag;
} __pygodot___Wrapped;

namespace pygodot {

class PyGodot {

public:
	static void set_pythonpath(godot_gdnative_init_options *o);
	static void python_init();
	static void python_terminate();

  static void nativescript_init(void *handle);
  static void nativescript_terminate(void *handle);

  static void register_pyscript_language();
};

} // namespace pygodot

#endif
