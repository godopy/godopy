#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"

#include "CoreTypes.hpp"

#include <wchar.h>

extern "C" PyObject *cython_nativescript_init();
extern "C" PyObject *cython_nativescript_terminate();
extern "C" PyObject *python_nativescript_init();
extern "C" PyObject *python_nativescript_terminate();

namespace pygodot {

void PyGodot::python_preconfig(godot_gdnative_init_options *options) {
	PyPreConfig preconfig;
	PyPreConfig_InitIsolatedConfig(&preconfig);

	preconfig.utf8_mode = 1;

	PyStatus status = Py_PreInitialize(&preconfig);

	if (PyStatus_Exception(status)) {
		FATAL_PRINT("Python Pre-Initialization Failed.");
		Py_ExitStatusException(status);
	}
}

#define ERR_FAIL_PYSTATUS(status, label) if (PyStatus_Exception(status)) goto label

void PyGodot::python_init() {
	PyStatus status;
	PyConfig config;

	status = PyConfig_InitIsolatedConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	// TODO: Set the full path of the real executable
	status = PyConfig_SetString(&config, &config.program_name, L"godot");
	ERR_FAIL_PYSTATUS(status, fail);

	// TODO: Set config.prefix from project path
	// TODO: Set other prefixes, executable and argv from Godot OS singleton

	status = PyConfig_SetString(&config, &config.pycache_prefix, L".__pycache__");
	ERR_FAIL_PYSTATUS(status, fail);

#ifdef _WIN32
	status = PyWideStringList_Append(&config.module_search_paths, L"C:\\demos\\cython-example\\pygodot");
	ERR_FAIL_PYSTATUS(status, fail);
	status = PyWideStringList_Append(&config.module_search_paths, L"C:\\demos\\cython-example\\pygodot\\deps\\python\\PCBuild\\amd64");
	ERR_FAIL_PYSTATUS(status, fail);
	status = PyWideStringList_Append(&config.module_search_paths, L"C:\\demos\\cython-example\\pygodot\\deps\\python\\Lib");
	ERR_FAIL_PYSTATUS(status, fail);
#elif __APPLE__
	status = PyWideStringList_Append(&config.module_search_paths, L"/Users/ii/src/pygodot");
	ERR_FAIL_PYSTATUS(status, fail);
	status = PyWideStringList_Append(&config.module_search_paths, L"/Users/ii/src/pygodot/buildenv/lib/python3.8");
	ERR_FAIL_PYSTATUS(status, fail);
	status = PyWideStringList_Append(&config.module_search_paths, L"/Users/ii/src/pygodot/buildenv/lib/python3.8/lib-dynload");
	ERR_FAIL_PYSTATUS(status, fail);
#else
	status = PyWideStringList_Append(&config.module_search_paths, L"/home/ii/src/pygodot");
	ERR_FAIL_PYSTATUS(status, fail);
	status = PyWideStringList_Append(&config.module_search_paths, L"/home/ii/src/pygodot/buildenv/lib/python3.8");
	ERR_FAIL_PYSTATUS(status, fail);
	status = PyWideStringList_Append(&config.module_search_paths, L"/home/ii/src/pygodot/buildenv/lib/python3.8/lib-dynload");
	ERR_FAIL_PYSTATUS(status, fail);
#endif

	config.isolated = 1;
	config.site_import = 0;
	config.buffered_stdio = 1;
	config.module_search_paths_set = 1;

	status = PyConfig_Read(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	// TODO: Override values computed by PyConfig_Read()

	status = Py_InitializeFromConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	PyConfig_Clear(&config);

	godot::Godot::print("Python {0}\nPyGodot 0.0.1a0\n", (godot::Variant)Py_GetVersion());
	return;

fail:
	FATAL_PRINT("Python Initialization Failed.");
	PyConfig_Clear(&config);
	Py_ExitStatusException(status);
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
