#include <GodotGlobal.hpp>
#include <PythonGlobal.hpp>

/* Generic gdnlib, users may create their own */

PyMODINIT_FUNC PyInit__pygodot(void);
PyMODINIT_FUNC PyInit_gdnative(void);
PyMODINIT_FUNC PyInit_nodes(void);
PyMODINIT_FUNC PyInit_utils(void);

// extern "C" void register_class(PyObject *);

extern "C" int _nativesctipt_python_init();
extern "C" int _gdnative_python_singleton();

extern "C" void GDN_EXPORT godot_gdnative_init(godot_gdnative_init_options *o) {
  godot::Godot::gdnative_init(o);
  pygodot::PyGodot::set_pythonpath(o);
}

extern "C" void GDN_EXPORT godot_gdnative_terminate(godot_gdnative_terminate_options *o) {
  godot::Godot::gdnative_terminate(o);
}

extern "C" void GDN_EXPORT godot_nativescript_init(void *handle) {
  PyImport_AppendInittab("_pygodot", PyInit__pygodot);
  PyImport_AppendInittab("gdnative", PyInit_gdnative);
  PyImport_AppendInittab("nodes", PyInit_nodes);
  PyImport_AppendInittab("utils", PyInit_utils);

  // Add "PyImport_AppendInittab"-s for custom modules here (which included within the binary extension)

  pygodot::PyGodot::python_init();

  // Importing of Cython modules is required to correctly initialize them
  PyObject *mod = NULL;
  mod = PyImport_ImportModule("gdnative"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("nodes"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("utils"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);

  // Import custom modules here

  pygodot::PyGodot::nativescript_init(handle);

  // Register custom NativeScript classes here

  // The rest of the function is needed to register Python modules dynamically and can be removed if
  // all required NativeScript modules are included in the binary
  if (_nativesctipt_python_init() != 0) PyErr_Print();
}

extern "C" void GDN_EXPORT godot_gdnative_singleton() {
  if (_gdnative_python_singleton() != 0) PyErr_Print();
}

extern "C" void GDN_EXPORT godot_nativescript_terminate(void *handle) {
  pygodot::PyGodot::nativescript_terminate(handle);
  pygodot::PyGodot::python_terminate();
}
