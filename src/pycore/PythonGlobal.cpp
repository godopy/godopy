#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"

#include <wchar.h>

extern "C" PyObject *cython_nativescript_init();
extern "C" PyObject *cython_nativescript_terminate();
extern "C" PyObject *python_nativescript_init();
extern "C" PyObject *python_nativescript_terminate();

namespace pygodot {

void PyGodot::set_pythonpath(godot_gdnative_init_options *options) {
  // TODO: Rename to python_preconfig(), use PyPreConfig_InitIsolatedConfig()
}

void PyGodot::python_init() {
  // TODO: PyConfig_InitIsolatedConfig()

  Py_NoUserSiteDirectory = 1;
  Py_NoSiteFlag = 1;
	// Py_IgnoreEnvironmentFlag = 1;

  // XXX: hardcoded
	Py_SetProgramName(L"godot");
#ifdef _WIN32
  Py_SetPath(
    L"C:\\demos\\cython-example\\pygodot"
    L";C:\\demos\\cython-example\\pygodot\\deps\\python\\PCBuild\\amd64"
    L";C:\\demos\\cython-example\\pygodot\\deps\\python\\Lib"
  );
#elif __APPLE__
	// Py_SetPythonHome(L"/Users/ii/src/pygodot/buildenv");
  printf("set path\n");
  Py_SetPath(
    L"/Users/ii/src/pygodot"
    L":/Users/ii/src/pygodot/buildenv/lib/python3.8"
    L":/Users/ii/src/pygodot/buildenv/lib/python3.8/lib-dynload"
  );
#else

#endif
	Py_InitializeEx(0);

  printf("Python %s\n\n", Py_GetVersion());
}

void PyGodot::python_terminate() {
	if (Py_IsInitialized()) {
		Py_FinalizeEx();
	}
}

void PyGodot::nativescript_init(void *handle, bool init_cython, bool init_python) {
	godot::_RegisterState::nativescript_handle = handle;

  if (init_cython) {
    if (cython_nativescript_init() == NULL) {
      if (PyErr_Occurred()) PyErr_Print();
      return;
    }
  }

  if (init_python) {
    if (python_nativescript_init() == NULL) {
      if (PyErr_Occurred()) PyErr_Print();
      return;
    }
  }
}

void PyGodot::nativescript_terminate(void *handle, bool terminate_cython, bool terminate_python) {
  if (terminate_cython) {
    if (cython_nativescript_terminate() == NULL) {
      if (PyErr_Occurred()) PyErr_Print();
    }
  }
  if (terminate_python) {
    if (python_nativescript_terminate() == NULL) {
      if (PyErr_Occurred()) PyErr_Print();
    }
  }
	godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::python_language_index);
}

void PyGodot::set_cython_language_index(int language_index) {
  godot::_RegisterState::cython_language_index = language_index;
}

void PyGodot::set_python_language_index(int language_index) {
  godot::_RegisterState::python_language_index = language_index;
}

} // namespace pygodot
