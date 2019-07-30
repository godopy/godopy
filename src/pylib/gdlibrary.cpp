/* Default generic gdnlib, may be replaced by custom builds */

#include <PyGodot.hpp>

#include "godot/bindings/_cython_bindings.h"
#include "godot/bindings/_python_bindings.h"
#include "godot/nativescript.h"
#include "godot/gdnative.h"
#include "pygodot/utils.h"

static bool _pygodot_is_initialized = false;
static godot_gdnative_init_options _cached_options;

#ifndef PYGODOT_EXPORT
#include <array>
const std::string shelloutput(const char* cmd) {
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

  return result;
}
#endif


static void _ensure_pygodot_is_initialized() {
  if (_pygodot_is_initialized) return;

#ifndef PYGODOT_EXPORT
  static bool use_pipenv = (system("python -c 'import pygodot;_=print' &> /dev/null") != 0);

  if (system(use_pipenv ? "pipenv run python -c 'import pygodot;_=print' &> /dev/null" :
                                     "python -c 'import pygodot;_=print' &> /dev/null") != 0) {
    throw std::runtime_error("unusable Python environment");
  }

  // Copy the correct Python paths for development
  const std::string dev_python_path = use_pipenv ?
    shelloutput("pipenv run python -c \"print(':'.join(__import__('sys').path))\"") :
    shelloutput("python -c \"print(':'.join(__import__('sys').path))\"") ;
  const wchar_t *w_dev_python_path = Py_DecodeLocale(dev_python_path.c_str(), NULL);
  Py_SetPath(w_dev_python_path);
  PyMem_RawFree((void *)w_dev_python_path);
#endif

  PyImport_AppendInittab("_pygodot", PyInit__pygodot);
  // XXX: Cython bindings are not needed in a generic lib
  PyImport_AppendInittab("_cython_bindings", PyInit__cython_bindings);
  PyImport_AppendInittab("_python_bindings", PyInit__python_bindings);
  PyImport_AppendInittab("utils", PyInit_utils);
  PyImport_AppendInittab("nativescript", PyInit_nativescript);
  PyImport_AppendInittab("gdnative", PyInit_gdnative);

  pygodot::PyGodot::python_init();

  // Importing of Cython modules is required to correctly initialize them
  PyObject *mod = NULL;
  // TODO: add Godot error handling
  // XXX: Cython bindings are not needed in a generic lib
  mod = PyImport_ImportModule("_cython_bindings"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("_python_bindings"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("utils"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("nativescript"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);
  mod = PyImport_ImportModule("gdnative"); if (mod == NULL) return PyErr_Print(); Py_DECREF(mod);

  _pygodot_is_initialized = true;
}

extern "C" void GDN_EXPORT pygodot_gdnative_init(godot_gdnative_init_options *o) {
    godot::Godot::gdnative_init(o);
    pygodot::PyGodot::set_pythonpath(o);

    _cached_options = *o;
}

extern "C" void GDN_EXPORT pygodot_gdnative_terminate(godot_gdnative_terminate_options *o) {
  godot::Godot::gdnative_terminate(o);
  pygodot::PyGodot::python_terminate();
}

extern "C" void GDN_EXPORT pygodot_nativescript_init(void *handle) {
  _ensure_pygodot_is_initialized();
  pygodot::PyGodot::nativescript_init(handle);

  if (_generic_pygodot_nativescript_init(_cached_options) != GODOT_OK) PyErr_Print(); // TODO: add Godot error handling
}

extern "C" void GDN_EXPORT pygodot_gdnative_singleton() {
  _ensure_pygodot_is_initialized();
  if (_generic_pygodot_gdnative_singleton() != GODOT_OK) PyErr_Print(); // TODO: add Godot error handling
}

extern "C" void GDN_EXPORT pygodot_nativescript_terminate(void *handle) {
  pygodot::PyGodot::nativescript_terminate(handle);
}
