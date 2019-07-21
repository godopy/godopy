#include <PyGodot.hpp>
#include "pygodot/gdnative.h"
#include "pygodot/nodes.h"
#include "pygodot/utils.h"
#include "pygodot/pyscript.h"

#include <array>

/* Default generic gdnlib, may be replaced by custom builds */

static bool _pygodot_is_initialized = false;
static bool _gdnative_is_initialized = false;

const wchar_t *shelloutput(const char* cmd) {
  std::array<char, 1024> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  result.erase(std::find_if(result.rbegin(), result.rend(), [](int ch) {
    return !std::isspace(ch);
  }).base(), result.end());

  return Py_DecodeLocale(result.c_str(), NULL);
}

static void _ensure_pygodot_is_initialized() {
  if (_pygodot_is_initialized) return;

#ifndef PYGODOT_EXPORT
  // TODO: Check Python version first
  // Copy active Python path for development
  const wchar_t *dev_python_path = shelloutput("python -c \"print(':'.join(__import__('sys').path))\"");
  Py_SetPath(dev_python_path);
  PyMem_RawFree((void *)dev_python_path);
#endif

  printf("__PYGODOT\n");

  PyImport_AppendInittab("_pygodot", PyInit__pygodot);
  PyImport_AppendInittab("nodes", PyInit_nodes);
  PyImport_AppendInittab("utils", PyInit_utils);
  PyImport_AppendInittab("gdnative", PyInit_gdnative);
  PyImport_AppendInittab("pyscript", PyInit_pyscript);

  // Add "PyImport_AppendInittab"-s for custom modules here (which included within the binary extension)

  pygodot::PyGodot::python_init();

  // Importing of Cython modules is required to correctly initialize them
  PyObject *mod = NULL;
  // TODO: add Godot error handling
  mod = PyImport_ImportModule("nodes"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("utils"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("gdnative"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("pyscript"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);

   // Import custom modules here

  _pygodot_is_initialized = true;
}

extern "C" void GDN_EXPORT pygodot_gdnative_init(godot_gdnative_init_options *o) {
  printf("PYGODOT GDNATIVE INIT\n");
  if (!_gdnative_is_initialized) {
    godot::Godot::gdnative_init(o);
    pygodot::PyGodot::set_pythonpath(o);
    _gdnative_is_initialized = true;
  }
}
extern "C" void GDN_EXPORT pyscript_gdnative_init(godot_gdnative_init_options *o) {
  printf("PYSCRIPT GDNATIVE INIT\n");
  if (!_gdnative_is_initialized) pygodot_gdnative_init(o);
}

extern "C" void GDN_EXPORT pygodot_gdnative_terminate(godot_gdnative_terminate_options *o) {
  godot::Godot::gdnative_terminate(o);
  pygodot::PyGodot::python_terminate();
}
extern "C" void GDN_EXPORT pyscript_gdnative_terminate(godot_gdnative_terminate_options *o) {}

extern "C" void GDN_EXPORT pygodot_nativescript_init(void *handle) {
  printf("NS INIT\n");
  _ensure_pygodot_is_initialized();
  pygodot::PyGodot::nativescript_init(handle);

  if (_nativescript_python_init() != 0) return PyErr_Print(); // TODO: add Godot error handling

  if (_register_class(_clsdef_PyGodotGlobal) != 0) PyErr_Print(); // TODO: add Godot error handling
  // Register custom NativeScript classes here
}

extern "C" void GDN_EXPORT pygodot_gdnative_singleton() {
  printf("PYGODOT SINGLETON\n");
  // _ensure_pygodot_is_initialized();
  // if (_gdnative_python_singleton() != 0) PyErr_Print(); // TODO: add Godot error handling
}

extern "C" void GDN_EXPORT pyscript_gdnative_singleton() {
  printf("PYSCRIPT SINGLETON\n");
  _ensure_pygodot_is_initialized();
  pygodot::PyGodot::register_pyscript_language();
  if (_gdnative_python_singleton() != 0) PyErr_Print(); // TODO: add Godot error handling
}

extern "C" void GDN_EXPORT pygodot_nativescript_terminate(void *handle) {
  pygodot::PyGodot::nativescript_terminate(handle);
}
