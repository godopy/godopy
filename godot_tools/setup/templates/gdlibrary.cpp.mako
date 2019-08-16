/* Generated by PyGodot ${__version__} */
#include "PythonGlobal.hpp"
#include "GodotGlobal.hpp"
#include "Defs.hpp"

#include <internal-packages/godot/nativescript.hpp>
#include <internal-packages/godot/gdnative.hpp>


PyMODINIT_FUNC PyInit_godot__core___wrapped();
PyMODINIT_FUNC PyInit_godot__core__signal_arguments();
PyMODINIT_FUNC PyInit_godot__core__tag_db();
PyMODINIT_FUNC PyInit_godot__core__wrapper_types();
PyMODINIT_FUNC PyInit_godot__bindings___cython_bindings();
PyMODINIT_FUNC PyInit_godot__bindings___python_bindings();
PyMODINIT_FUNC PyInit_godot__utils();

% for mod in pyx_sources:
PyMODINIT_FUNC PyInit_${mod['symbol_name']}(void);
% endfor

extern "C" PyObject *_pygodot_nativescript_init();
% if singleton:
extern "C" PyObject *_pygodot_gdnative_singleton();
% endif
% if gdnative_init:
extern "C" PyObject *_pygodot_gdnative_init(godot_gdnative_init_options *)
% endif

static bool __python_initialized = false;

static PyModuleDef ${library_name}module = {
  PyModuleDef_HEAD_INIT, "${library_name}",
  "${repr(library_name)} GDNative extension",
  -1
};

PyMODINIT_FUNC PyInit_${library_name}(void) { return PyModule_Create(&${library_name}module); }

static void ___python_init() {
  if (__python_initialized) return;

  PyImport_AppendInittab("${library_name}", PyInit_${library_name});
  PyImport_AppendInittab("__pygodot_internal__godot__core___wrapped", PyInit_godot__core___wrapped);
  PyImport_AppendInittab("__pygodot_internal__godot__core__signal_arguments", PyInit_godot__core__signal_arguments);
  PyImport_AppendInittab("__pygodot_internal__godot__core__tag_db", PyInit_godot__core__tag_db);
  PyImport_AppendInittab("__pygodot_internal__godot__core__wrapper_types", PyInit_godot__core__wrapper_types);
  PyImport_AppendInittab("__pygodot_internal__godot__bindings___cython_bindings", PyInit_godot__bindings___cython_bindings);
  PyImport_AppendInittab("__pygodot_internal__godot__bindings___python_bindings", PyInit_godot__bindings___python_bindings);
  PyImport_AppendInittab("__pygodot_internal__godot__utils", PyInit_godot__utils);
  PyImport_AppendInittab("__pygodot_internal__godot__nativescript", PyInit_godot__nativescript);
  PyImport_AppendInittab("__pygodot_internal__godot__gdnative", PyInit_godot__gdnative);

% for mod in reversed(pyx_sources):
  PyImport_AppendInittab("__pygodot_internal__${mod['symbol_name']}", PyInit_${mod['symbol_name']});
% endfor
  pygodot::PyGodot::python_init();

  PyObject *mod = NULL;

  // Importing of Cython modules is required to correctly initialize them
  mod = PyImport_ImportModule("__pygodot_internal__godot__core___wrapped"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__core__signal_arguments"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__core__tag_db"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__core__wrapper_types"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__bindings___cython_bindings"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__bindings___python_bindings"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__utils"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__nativescript"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
  mod = PyImport_ImportModule("__pygodot_internal__godot__gdnative"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);

% for mod in reversed(pyx_sources):
  mod = PyImport_ImportModule("__pygodot_internal__${mod['symbol_name']}"); ERR_FAIL_PYTHON_NULL(mod); Py_DECREF(mod);
% endfor

  __python_initialized = true;
}

extern "C" void GDN_EXPORT pygodot_gdnative_init(godot_gdnative_init_options *o) {
  godot::Godot::gdnative_init(o);
  pygodot::PyGodot::python_preconfig(o);

% if gdnative_init:
  PyObject *result = _pygodot_gdnative_init(o); ERR_FAIL_PYTHON_NULL(result);
% endif
}

extern "C" void GDN_EXPORT pygodot_gdnative_terminate(godot_gdnative_terminate_options *o) {
  godot::Godot::gdnative_terminate(o);
  pygodot::PyGodot::python_terminate();
}

extern "C" void GDN_EXPORT pygodot_nativescript_init(void *handle) {
  printf("NATIVESCRIPT INIT\n");
  godot::Godot::nativescript_init(handle);  // C++ bindings
  ___python_init();
  pygodot::PyGodot::nativescript_init(handle);

  PyObject *result = _pygodot_nativescript_init(); ERR_FAIL_PYTHON_NULL(result);
}

extern "C" void GDN_EXPORT pygodot_gdnative_singleton() {
  ___python_init();
% if singleton:
  PyObject *result = _pygodot_gdnative_singleton(); ERR_FAIL_PYTHON_NULL(result);
% endif
}

extern "C" void GDN_EXPORT pygodot_nativescript_terminate(void *handle) {
  pygodot::PyGodot::nativescript_terminate(handle);
  godot::Godot::nativescript_terminate(handle); // C++ bindings
}
