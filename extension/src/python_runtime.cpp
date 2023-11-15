#include "python_runtime.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/os.hpp>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyMODINIT_FUNC PyInit__godopy_bootstrap(void);

using namespace godot;

PythonRuntime *PythonRuntime::singleton = nullptr;

_FORCE_INLINE_ const wchar_t *_wide_string_from_string(String &s) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, convert to 16-bit
	return (const wchar_t *)s.utf16().ptr();
#else
	// wchar_t is 32-bit, return as is
	return (const wchar_t *)s.ptr();
#endif
}

void PythonRuntime::pre_initialize() {
	UtilityFunctions::print("Python: Pre-Initializing module...");

	PyPreConfig preconfig;
	PyPreConfig_InitIsolatedConfig(&preconfig);

	preconfig.utf8_mode = 1;
	preconfig.isolated = 1;
	preconfig.use_environment = 0;

	PyStatus status = Py_PreInitialize(&preconfig);

	if (PyStatus_Exception(status)) {
		ERR_PRINT("Python Pre-Initialization Failed.");
		Py_ExitStatusException(status);
	}
}

#define ERR_FAIL_PYSTATUS(status, label) if (PyStatus_Exception(status)) goto label
#define CHECK_PYSTATUS(status, ret) if (PyStatus_Exception(status)) return ret

int set_config_paths(PyConfig *config) {
	ProjectSettings *settings = ProjectSettings::get_singleton();

	String res_path = settings->globalize_path("res://");
	// String godot_project_path = settings->globalize_path("res://project.godot");
	String project_name = settings->get_setting("application/config/name");

	if (project_name == "") {
		ERR_PRINT("No Godot/GodoPy project found. Cannot run.");
		return 1;
	}

	// String godopy_root = res_path.get_base_dir();

	UtilityFunctions::print("Python path: " + res_path + "../python/Lib");

	PyStatus status;
	String exec_path = OS::get_singleton()->get_executable_path();
	// List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();

	String exec_prefix = exec_path.get_base_dir().get_base_dir();

	UtilityFunctions::print("Python program name: " + exec_path);

	status = PyConfig_SetString(config, &config->program_name, _wide_string_from_string(exec_path));
	CHECK_PYSTATUS(status, 1);

	//for (List<String>::Element *E = cmdline_args.front(); E; E = E->next()) {
	//	status = PyWideStringList_Append(&config->argv, _wide_string_from_string(E->get()));
	//	CHECK_PYSTATUS(status, 1);
	//}

	status = PyConfig_SetString(config, &config->base_exec_prefix, _wide_string_from_string(exec_prefix));
	CHECK_PYSTATUS(status, 1);

	status = PyConfig_SetString(config, &config->base_prefix, _wide_string_from_string(res_path));
	CHECK_PYSTATUS(status, 1);

	status = PyConfig_SetString(config, &config->exec_prefix, _wide_string_from_string(exec_prefix));
	CHECK_PYSTATUS(status, 1);

	status = PyConfig_SetString(config, &config->executable, _wide_string_from_string(exec_path));
	CHECK_PYSTATUS(status, 1);

	status = PyConfig_SetString(config, &config->prefix, _wide_string_from_string(res_path));
	CHECK_PYSTATUS(status, 1);

	// TODO: Copy Python libs and dylibs to project/addons/GodoPy
	status = PyWideStringList_Append(&config->module_search_paths, _wide_string_from_string(res_path + "../python/Lib"));
	CHECK_PYSTATUS(status, 1);

	return 0;
}

void init_builtin_modules() {
	PyImport_AppendInittab("_godopy_bootstrap", PyInit__godopy_bootstrap);
}

void PythonRuntime::initialize() {
	UtilityFunctions::print("Python: Initializing runtime...");
	UtilityFunctions::print("Python version " + String(Py_GetVersion()));

	PyStatus status;
	PyConfig config;

	init_builtin_modules();

	PyConfig_InitIsolatedConfig(&config);

	if (set_config_paths(&config) != 0) {
		goto fail;
	}

	config.verbose = 0;
	config.isolated = 1;
	config.site_import = 0;
	config.faulthandler = 0;
	config.buffered_stdio = 1;
	config.write_bytecode = 1;
	config.use_environment = 0;
	config.user_site_directory = 0;
	config.install_signal_handlers = 0;
	config.module_search_paths_set = 1;

	status = PyConfig_Read(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	status = Py_InitializeFromConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	initialized = true;

	PyConfig_Clear(&config);

	// Redirect stdio
	PyRun_SimpleString("import _godopy_bootstrap");

	UtilityFunctions::print("Python: INITIALIZED");

	// TODO: print banners

	return;

fail:
	PyConfig_Clear(&config);
	UtilityFunctions::print("Python: Initialization FAILED");
	Py_ExitStatusException(status);
}

void PythonRuntime::run_simple_string(const String &p_string) {
	ERR_FAIL_COND(!initialized);
	PyRun_SimpleString(p_string.utf8());
}

PythonRuntime::PythonRuntime() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	initialized = false;
}

PythonRuntime::~PythonRuntime() {
	if (is_initialized()) {
		if (Py_IsInitialized()) {
			Py_FinalizeEx();
		}
		initialized = false;
	}

	ERR_FAIL_COND(singleton != this);
	singleton = nullptr;
}


Python *Python::singleton = nullptr;


void Python::run_simple_string(const String &p_string) {
	PythonRuntime::get_singleton()->run_simple_string(p_string);
}

void Python::_bind_methods() {
	//ClassDB::bind_method(D_METHOD("is_initialized"), &PythonRuntime::is_initialized);
	ClassDB::bind_method(D_METHOD("run_simple_string", "string"), &Python::run_simple_string);
	//ClassDB::bind_method(D_METHOD("run_headless_shell"), &PythonRuntime::run_headless_shell);
}

Python::Python() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

Python::~Python() {
	ERR_FAIL_COND(singleton != this);
	singleton = nullptr;
}
