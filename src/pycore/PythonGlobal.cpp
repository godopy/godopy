#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"

#include "CoreTypes.hpp"

#include <OS.hpp>
#include <ProjectSettings.hpp>

#ifdef _WIN32
// placeholder for Windows includes
#elif __APPLE__
// placeholder for macOS/iOS includes
#else
#include <dlfcn.h>
#endif

extern "C" PyObject *cython_nativescript_init();
extern "C" PyObject *cython_nativescript_terminate();
extern "C" PyObject *python_nativescript_init();
extern "C" PyObject *python_nativescript_terminate();
extern "C" PyObject *global_nativescript_terminate();

namespace godopy {

bool in_editor = false;
godot_string active_library_path;

void GodoPy::python_preconfig(godot_gdnative_init_options *options) {
	PyPreConfig preconfig;
	PyPreConfig_InitIsolatedConfig(&preconfig);

	preconfig.utf8_mode = 1;

	PyStatus status = Py_PreInitialize(&preconfig);

	if (PyStatus_Exception(status)) {
		FATAL_PRINT("Python Pre-Initialization Failed.");
		Py_ExitStatusException(status);
	}

	godot::api->godot_string_new_copy(&active_library_path, options->active_library_path);
	in_editor = options->in_editor;
}

#define ERR_FAIL_PYSTATUS(status, label) if (PyStatus_Exception(status)) goto label

void GodoPy::python_init() {
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

	if (!settings->has_setting("python/config/module_search_path/main")) {
		FATAL_PRINT("Python Initialization Failed: no Python module search path defined.");
		CRASH_NOW();
		return;
	}

	godot::String main_module_path = settings->globalize_path(settings->get_setting("python/config/module_search_path/main"));
	godot::String tool_module_path = settings->has_setting("python/config/module_search_path/extended") ?
		settings->globalize_path(settings->get_setting("python/config/module_search_path/extended")) : "";

	godot::String development_module_path = settings->has_setting("python/config/module_search_path/development") ?
		settings->get_setting("python/config/module_search_path/development") : "";

	godot::String binary_module_path = ((godot::String *)&active_library_path)->get_base_dir();

	bool commandline_script_mode = false;

	bool write_bytecode = false;

#ifdef _WIN32

#elif __APPLE__

#else
	// printf("dlopen %s\n", ((godot::String *)&active_library_path)->utf8().get_data());

	// Make Python symbols available for core Python extension modules on Linux
	// Idea from https://stackoverlow.com/questions/11842920/undefined-symbol-pyexc-importerror-when-embedding-python-in-c#11847653
	const void *___so_handle = dlopen(((godot::String *)&active_library_path)->utf8().get_data(), RTLD_LAZY | RTLD_GLOBAL);
#endif
	godot::api->godot_string_destroy(&active_library_path);

	PyConfig_InitIsolatedConfig(&config);

	status = PyConfig_SetString(&config, &config.program_name, executable.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	for (int i = 0; i < _argv.size(); i++) {
		if (_argv[i] == "-s") {
			commandline_script_mode = true;
		}
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

	if (development_module_path != "") {
		// godot::Godot::print("set DEV Python path: {0}", development_module_path);
		status = PyWideStringList_Append(&config.module_search_paths, development_module_path.unicode_str());
		ERR_FAIL_PYSTATUS(status, fail);

		write_bytecode = true;
	}

	// godot::Godot::print("set MAIN Python path: {0}", main_module_path);
	status = PyWideStringList_Append(&config.module_search_paths, main_module_path.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	if (tool_module_path != "" && (in_editor || commandline_script_mode || development_module_path != "")) {
		// godot::Godot::print("set TOOL python path: {0}", tool_module_path);
		status = PyWideStringList_Append(&config.module_search_paths, tool_module_path.unicode_str());
		ERR_FAIL_PYSTATUS(status, fail);
	}

	// godot::Godot::print("set BIN python path: {0}", binary_module_path);
	status = PyWideStringList_Append(&config.module_search_paths, binary_module_path.unicode_str());
	ERR_FAIL_PYSTATUS(status, fail);

	config.verbose = 0;
	config.isolated = 1;
	config.site_import = 0;
	config.faulthandler = 0;
	config.buffered_stdio = 1;
	config.write_bytecode = write_bytecode;  // TODO: Enable bytecode if possible, set pycache_prefix
	config.use_environment = 0;
	config.user_site_directory = 0;
	config.install_signal_handlers = 0;
	config.module_search_paths_set = 1;

	status = PyConfig_Read(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	status = Py_InitializeFromConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	PyConfig_Clear(&config);

	// Initializes the GIL, required for threading
	PyEval_InitThreads();

	// Python can run in a multithreaded environment.  Limitations:
	// * Pure Python code is single threaded, it can't run concurently in multiple threads.
	// * Something must release the GIL in its "_process" method.  Otherwise threaded code
	//   may not have a chance to execute.  Cython syntax is "with nogil:". Even "with nogil: pass" will do the trick
	// Non-limitations:
	// * Python code runs concurently with non-Python code without issues

	if (!commandline_script_mode) {
		godot::Godot::print("Python {0}\nGodoPy 0.0.1\n", (godot::Variant)Py_GetVersion());
	}

	return;

fail:
	FATAL_PRINT("Python Initialization Failed.");
	PyConfig_Clear(&config);
	Py_ExitStatusException(status);
}

void GodoPy::python_terminate() {
	if (Py_IsInitialized()) {

		Py_FinalizeEx();
	}
}

void GodoPy::numpy_init() {
	if (_import_array() == -1) {
		PyErr_Print();
		FATAL_PRINT("NumPy Initialization Failed.");
		CRASH_NOW();
		return;
	}
}

void GodoPy::nativescript_init(void *handle, bool init_cython, bool init_python) {
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

void GodoPy::nativescript_terminate(void *handle, bool terminate_cython, bool terminate_python) {
	if (terminate_python) {
		if (python_nativescript_terminate() == NULL) {
			if (PyErr_Occurred()) PyErr_Print();
		}

		godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::python_language_index);
	}

	if (terminate_cython) {
		if (cython_nativescript_terminate() == NULL) {
			if (PyErr_Occurred()) PyErr_Print();
		}

		godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::cython_language_index);
	}

	if (global_nativescript_terminate() == NULL) {
		if (PyErr_Occurred()) PyErr_Print();
	}
}

void GodoPy::set_cython_language_index(int language_index) {
	godot::_RegisterState::cython_language_index = language_index;
}

void GodoPy::set_python_language_index(int language_index) {
	godot::_RegisterState::python_language_index = language_index;
}

} // namespace godopy
