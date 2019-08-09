#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"

#include "CoreTypes.hpp"

#include "OS.hpp"
#include "ProjectSettings.hpp"

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
	// In singleton configuration this function is called before 'nativescript_init' and method bindings are not set
	godot::OS::___init_method_bindings();
	godot::ProjectSettings::___init_method_bindings();

	godot::OS *os = godot::OS::get_singleton();
	godot::ProjectSettings *settings = godot::ProjectSettings::get_singleton();

	PyStatus status;
	PyConfig config;

	// godot::Godot::print("get_user_data_dir: {0}", os->get_user_data_dir());
	// godot::Godot::print("get_locale: {0}", os->get_locale());
	// godot::Godot::print("get_model_name: {0}", os->get_model_name());
	// godot::Godot::print("get_name: {0}", os->get_name());

	godot::String executable = os->get_executable_path();
	godot::String exec_prefix = executable.get_base_dir().get_base_dir();
	godot::String project = settings->globalize_path("res://project.godot");
	godot::String prefix = project.get_base_dir();
	godot::PoolStringArray _argv = os->get_cmdline_args();

	status = PyConfig_InitIsolatedConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	status = PyConfig_SetString(&config, &config.program_name, executable.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	for (int i = 0; i < _argv.size(); i++) {
		status = PyWideStringList_Append(&config.argv, _argv[i].unicode_str());
		ERR_FAIL_PYSTATUS(status, fail);
	}

	status = PyConfig_SetString(&config, &config.base_exec_prefix, exec_prefix.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	status = PyConfig_SetString(&config, &config.base_prefix, prefix.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	status = PyConfig_SetString(&config, &config.exec_prefix, exec_prefix.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	status = PyConfig_SetString(&config, &config.executable, executable.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	status = PyConfig_SetString(&config, &config.prefix, prefix.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	// TODO: Set pycache_prefix in an user_data_dir subfolder if it is writable
	// status = PyConfig_SetString(&config, &config.pycache_prefix, L"");
	// ERR_FAIL_PYSTATUS(status, fail);

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

	config.verbose = 0;
	config.isolated = 1;
	config.site_import = 0;
	config.faulthandler = 0;
	config.buffered_stdio = 1;
	config.write_bytecode = 0;  // TODO: Enable bytecode if possible, set pycache_prefix
	config.use_environment = 0;
	config.user_site_directory = 0;
	config.install_signal_handlers = 0;
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
