#include "PythonGlobal.hpp"

#include <wchar.h>

namespace godot {

wchar_t *pythonpath = nullptr;
const godot_gdnative_core_api_struct *api = nullptr;

void Python::set_pythonpath(godot_gdnative_init_options *options) {
	api = options->api_struct;

	godot_string path = api->godot_string_get_base_dir(options->active_library_path);
	godot_int size = api->godot_string_length(&path);

	pythonpath = (wchar_t *)PyMem_RawMalloc((size + 7) * sizeof(wchar_t));
	wcsncpy(pythonpath, api->godot_string_wide_str(&path), size);
	wcsncpy(pythonpath + size, L"/pyres", 7);

	api->godot_string_destroy(&path);
}

void Python::init() {
	if (!pythonpath) {
		printf("Could not initialize Python interpreter:\n");
		printf("Python path was not defined!\n");

		return;
	}

	const char *c_pythonpath = Py_EncodeLocale(pythonpath, nullptr);

	Py_NoSiteFlag = 1;
	Py_IgnoreEnvironmentFlag = 1;

	Py_SetProgramName(L"godot");
	Py_SetPythonHome(pythonpath);

	// Initialize interpreter but skip initialization registration of signal handlers
	Py_InitializeEx(0);

	PyObject *mod = PyImport_ImportModule("godot");
  if (mod != NULL) {
    Py_DECREF(mod);

    printf("Python %s\n\n", Py_GetVersion());
  } else {
    PyErr_Print();
  }
}

void Python::terminate() {
	if (Py_IsInitialized()) {
		Py_FinalizeEx();

		if (pythonpath)
			PyMem_RawFree((void *)pythonpath);
	}
}

} // namespace godot
