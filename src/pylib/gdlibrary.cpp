#include <GodotGlobal.hpp>
#include <PythonGlobal.hpp>

/* Generic gdnlib, users may create their own */

#define LOADER_MODULE "__loader__"
#define LOADER_ENTRY_POINT "main"

PyMODINIT_FUNC PyInit_pygodot(void);
PyMODINIT_FUNC PyInit_gdnative(void);
PyMODINIT_FUNC PyInit_nodes(void);
PyMODINIT_FUNC PyInit_utils(void);

extern "C" void register_class(PyObject *);

extern "C" void GDN_EXPORT godot_gdnative_init(godot_gdnative_init_options *o) {
  godot::Godot::gdnative_init(o);
  pygodot::PyGodot::set_pythonpath(o);
}

extern "C" void GDN_EXPORT godot_gdnative_terminate(godot_gdnative_terminate_options *o) {
  godot::Godot::gdnative_terminate(o);
}

extern "C" void GDN_EXPORT godot_nativescript_init(void *handle) {
  PyImport_AppendInittab("pygodot", PyInit_pygodot);
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
  mod = PyImport_ImportModule(LOADER_MODULE);
  if (mod == NULL) {
    PyErr_Print();
    fprintf(stderr, "Failed to initialize PyGodot development mode.\n\"%s\" module not found.\n", LOADER_MODULE);
    return;
  }

  PyObject *main = PyObject_GetAttrString(mod, LOADER_ENTRY_POINT);

  if (main != NULL && PyCallable_Check(main)) {
    PyObject *arg = PyTuple_New(0);
    PyObject *result = PyObject_CallObject(main, arg);
    Py_XDECREF(arg);

    if (result == NULL || PyErr_Occurred()) PyErr_Print();
    Py_XDECREF(result);
  } else {
    Py_XDECREF(main);
    PyErr_Print();
    fprintf(stderr, "Failed to call \"%s.%s\"\n", LOADER_MODULE, LOADER_ENTRY_POINT);
  }

  Py_DECREF(mod);
}

extern "C" void GDN_EXPORT godot_nativescript_terminate(void *handle) {
  pygodot::PyGodot::nativescript_terminate(handle);
  pygodot::PyGodot::python_terminate();
}
