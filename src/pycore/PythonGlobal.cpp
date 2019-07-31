#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"

#include <wchar.h>

extern "C" PyObject *cython_nativescript_init();
extern "C" PyObject *python_nativescript_init();

namespace pygodot {

wchar_t *pythonpath = nullptr;

void PyGodot::set_pythonpath(godot_gdnative_init_options *options) {
	const godot_gdnative_core_api_struct *api = options->api_struct;

	godot_string dir = api->godot_string_get_base_dir(options->active_library_path);
  godot_string file = api->godot_string_get_file(options->active_library_path);
	godot_int dirsize = api->godot_string_length(&dir);
  godot_int filesize = api->godot_string_length(&file);

	pythonpath = (wchar_t *)PyMem_RawMalloc((dirsize + 1 + filesize + 5) * sizeof(wchar_t));
	wcsncpy(pythonpath, api->godot_string_wide_str(&dir), dirsize);
  wcsncpy(pythonpath + dirsize, L"/", 1);
  wcsncpy(pythonpath + dirsize + 1, api->godot_string_wide_str(&file), filesize);
	wcsncpy(pythonpath + dirsize + 1 + filesize, L".env", 5);

	api->godot_string_destroy(&dir);
  api->godot_string_destroy(&file);
}

void PyGodot::python_init() {
	if (!pythonpath) {
		printf("Could not initialize Python interpreter:\n");
		printf("Python path was not defined!\n");

		return;
	}

  Py_NoUserSiteDirectory = 1;

#ifdef PYGODOT_EXPORT
  Py_NoSiteFlag = 1;
	Py_IgnoreEnvironmentFlag = 1;
#endif

	Py_SetProgramName(L"godot");
	Py_SetPythonHome(pythonpath);

	// Initialize interpreter but skip initialization registration of signal handlers
	Py_InitializeEx(0);

	PyObject *mod = PyImport_ImportModule("pygodot");
  if (mod != NULL) {
    Py_DECREF(mod);

    printf("Python %s\n\n", Py_GetVersion());
  } else {
    PyErr_Print();
  }
}

void PyGodot::python_terminate() {
	if (Py_IsInitialized()) {
		Py_FinalizeEx();

		if (pythonpath)
			PyMem_RawFree((void *)pythonpath);
	}
}

void PyGodot::nativescript_init(void *handle, bool init_cython, bool init_python) {
	godot::_RegisterState::nativescript_handle = handle;

  if (init_cython) {
    if (cython_nativescript_init() == NULL) {
      PyErr_Print();
      return;
    }
  }

  if (init_python) {
    if (python_nativescript_init() == NULL) {
      PyErr_Print();
      return;
    }
  }
}

void PyGodot::nativescript_terminate(void *handle) {
	godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::python_language_index);
}

} // namespace pygodot
