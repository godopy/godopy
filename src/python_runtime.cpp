#include "python_runtime.h"

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/os.hpp>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyMODINIT_FUNC PyInit_godot(void);

using namespace godot;

PythonRuntime *PythonRuntime::singleton = nullptr;

_ALWAYS_INLINE_ const wchar_t *_wide_string_from_string(const String &s) {
	return s.wide_string().ptr();
}

void PythonRuntime::pre_initialize() {
	UtilityFunctions::print_verbose("Python: Pre-Initializing runtime...");

	PyPreConfig preconfig;
	PyPreConfig_InitIsolatedConfig(&preconfig);

	preconfig.utf8_mode = 1;

	PyStatus status = Py_PreInitialize(&preconfig);

	if (PyStatus_Exception(status)) {
		UtilityFunctions::push_error("Python: Pre-Initialization FAILED");
		Py_ExitStatusException(status);
	}
}

#define ERR_FAIL_PYSTATUS(status, label) if (PyStatus_Exception(status)) goto label
#define CHECK_PYSTATUS(status, ret) if (PyStatus_Exception(status)) return ret

int set_config_paths(PyConfig *config) {
	PyStatus status;
	ProjectSettings *settings = ProjectSettings::get_singleton();

	String res_path = settings->globalize_path("res://");
	String project_name = settings->get_setting("application/config/name");

	if (project_name == "") {
		ERR_PRINT("No Godot/GodoPy project found. Cannot run.");
		return 1;
	}

	// TODO: Get from project settings
	String exec_path = res_path + "/bin/windows/libGodoPy.dll";
	String exec_prefix = exec_path.get_base_dir();

	UtilityFunctions::print_verbose("Python program name: " + exec_path);

	status = PyConfig_SetString(config, &config->program_name, _wide_string_from_string(exec_path));
	CHECK_PYSTATUS(status, 1);

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

	status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path)
	);
	CHECK_PYSTATUS(status, 1);

	status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path + "pystdlib")
	);
	CHECK_PYSTATUS(status, 1);

	status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path + "pystdlib/site-packages")
	);
	CHECK_PYSTATUS(status, 1);

    status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path + "bin/windows/dylib")
	);
	CHECK_PYSTATUS(status, 1);

	// TODO: Detect active venv and editor/script status and set the right path
	//       only in editor or script
	status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path + "../../venv/Lib/site-packages")
	);
	CHECK_PYSTATUS(status, 1);

	// Also only for developing
	status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path + "../../python/Lib")
	);
	CHECK_PYSTATUS(status, 1);

	// Also only for developing
	status = PyWideStringList_Append(
		&config->module_search_paths,
		_wide_string_from_string(res_path + "../../python/PCbuild/amd64")
	);
	CHECK_PYSTATUS(status, 1);

	return 0;
}

void PythonRuntime::initialize() {
	UtilityFunctions::print_verbose("Python: Initializing runtime...");
	UtilityFunctions::print("Python version " + String(Py_GetVersion()));

	PyStatus status;
	PyConfig config;

	PyImport_AppendInittab("godot", PyInit_godot);

	PyConfig_InitIsolatedConfig(&config);

	if (set_config_paths(&config) != 0) {
		goto fail;
	}

	config.site_import = 0;
	// config.faulthandler = 0;
	// config.user_site_directory = 0;
	// config.install_signal_handlers = 0;
	config.module_search_paths_set = 1;

	// FIXME: custom faulthandler?

	status = PyConfig_Read(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	status = Py_InitializeFromConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	initialized = true;

	PyConfig_Clear(&config);

	return;

fail:
	PyConfig_Clear(&config);
	UtilityFunctions::push_error("Python: Initialization FAILED.");
	Py_ExitStatusException(status);
}

void PythonRuntime::run_simple_string(const String &p_string) {
	ERR_FAIL_COND(!initialized);
	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyRun_SimpleString(p_string.utf8());
	PyGILState_Release(gil_state);
}

PythonObject *PythonRuntime::import_module(const String &p_name) {
	PythonObject *module = memnew(PythonObject);
	module->set_name(p_name);

	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyObject *m = PyImport_ImportModule(p_name.utf8());
	ERR_FAIL_NULL_V(m, module);

    Py_INCREF(m);
    module->set_instance(m);
	PyObject *repr = PyObject_Repr(m);
    ERR_FAIL_NULL_V(repr, module);
    module->set_repr(String(repr));
    PyGILState_Release(gil_state);

    return module;
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

PythonObject *Python::import_module(const String &p_name) {
	return PythonRuntime::get_singleton()->import_module(p_name);
}

void Python::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run_simple_string", "string"), &Python::run_simple_string);
	ClassDB::bind_method(D_METHOD("import_module", "string"), &Python::import_module);
}

Python::Python() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

Python::~Python() {
	ERR_FAIL_COND(singleton != this);
	singleton = nullptr;
}
