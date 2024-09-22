#include "python_runtime.h"

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/engine.hpp>

PyMODINIT_FUNC PyInit__godopy_core(void);
PyMODINIT_FUNC PyInit__godot_types(void);

using namespace godot;

PythonRuntime *PythonRuntime::singleton = nullptr;

PythonRuntime::PythonRuntime() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	initialized = false;
	internal::gdextension_interface_get_library_path(internal::library, &library_path);
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

#define SET_PYCONFIG_STRING(config_param, path) \
	status = PyConfig_SetString(config, config_param, path); \
	if (PyStatus_Exception(status)) return 1

#define APPEND_PYTHON_PATH(path) \
	status = PyWideStringList_Append(&config->module_search_paths, path); \
	if (PyStatus_Exception(status)) return 1

int PythonRuntime::set_config_paths(PyConfig *config) {
	PyStatus status;

	String exec_path = library_path;
	String exec_prefix = exec_path.get_base_dir();

	String bin_dir = exec_prefix.get_base_dir();
	String res_path = bin_dir.get_base_dir();

	UtilityFunctions::print_verbose("Python library name: " + exec_path);
	UtilityFunctions::print_verbose("Detected project folder: " + res_path);

	SET_PYCONFIG_STRING(&config->program_name, exec_path.wide_string());
	SET_PYCONFIG_STRING(&config->base_exec_prefix, exec_prefix.wide_string());
	SET_PYCONFIG_STRING(&config->base_prefix, res_path.wide_string());
	SET_PYCONFIG_STRING(&config->exec_prefix, exec_prefix.wide_string());
	SET_PYCONFIG_STRING(&config->executable, exec_path.wide_string());
	SET_PYCONFIG_STRING(&config->prefix, res_path.wide_string());

	String python_lib_path = res_path.path_join("python").path_join("lib");

	APPEND_PYTHON_PATH(res_path.wide_string());
	APPEND_PYTHON_PATH(res_path.path_join("lib").wide_string());
	APPEND_PYTHON_PATH(python_lib_path.wide_string());
	APPEND_PYTHON_PATH(python_lib_path.path_join("site-packages").wide_string());
	APPEND_PYTHON_PATH(exec_prefix.path_join("dylib").wide_string());

	// Add venv to path only when editing
	// String venv_path = OS::get_singleton()->get_environment("VIRTUAL_ENV");
	// if (Engine::get_singleton()->is_editor_hint() && venv_path != "") {
	// 	APPEND_PYTHON_PATH(venv_path.path_join("Lib").path_join("site-packages").wide_string());
	// }

	return 0;
}

void PythonRuntime::initialize() {
	pre_initialize();

	UtilityFunctions::print_verbose("Python: Initializing runtime...");
	UtilityFunctions::print("Python version " + String(Py_GetVersion()));

	PyStatus status;
	PyConfig config;

	PyImport_AppendInittab("_godopy_core", PyInit__godopy_core);
	PyImport_AppendInittab("_godot_types", PyInit__godot_types);

	PyConfig_InitIsolatedConfig(&config);

	UtilityFunctions::print_verbose("Python: Configuring paths...");

	if (set_config_paths(&config) != 0) {
		goto fail;
	}

	config.site_import = 0;
	config.install_signal_handlers = 0;
	config.module_search_paths_set = 1;

	status = PyConfig_Read(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	UtilityFunctions::print_verbose("Python: Initializeing the interpreter...");
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

Ref<PythonObject> PythonRuntime::import_module(const String &p_name) {
	Ref<PythonObject> module = memnew(PythonObject);
	module->set_name(p_name);

	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyObject *m = PyImport_ImportModule(p_name.utf8());
	if (m == nullptr) {
		if (PyErr_Occurred()) {
			PyErr_Print();
			PyErr_Clear();
		}
		PyGILState_Release(gil_state);
		ERR_FAIL_NULL_V(m, module);
		return module;
	}
	ERR_FAIL_NULL_V(m, module);

    Py_INCREF(m);
    module->set_instance(m);
	PyObject *repr = PyObject_Repr(m);
    ERR_FAIL_NULL_V(repr, module);
    module->set_repr(String(repr));
    PyGILState_Release(gil_state);

    return module;
}

void PythonRuntime::finalize() {
	if (is_initialized()) {
		if (Py_IsInitialized()) {
			Py_FinalizeEx();
		}

		initialized = false;
	}
}

PythonRuntime::~PythonRuntime() {
	ERR_FAIL_COND(singleton != this);
	singleton = nullptr;
}


Python *Python::singleton = nullptr;

void Python::run_simple_string(const String &p_string) {
	PythonRuntime::get_singleton()->run_simple_string(p_string);
}

Ref<PythonObject> Python::import_module(const String &p_name) {
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
